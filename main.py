import base64
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from typing import Optional

import cv2
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address
from supabase import create_client, Client

from annotation import call_gemini_with_annotations, draw_annotations, get_image_dimensions

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("athletiq")

# ── App Setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AthletiQ API",
    description="AI-powered athlete performance analysis API",
    version="1.0.0",
)

# ── Rate limiting ──────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── CORS — restrict to your actual app domain(s) ───────────────────────────────
# Set ALLOWED_ORIGINS as a comma-separated env var, e.g. "https://yourapp.com,https://app.yourapp.com"
_allowed_origins = os.environ.get("ALLOWED_ORIGINS", "").split(",")
_allowed_origins = [o.strip() for o in _allowed_origins if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins or [],  # empty list = no browser origins allowed by default
    allow_methods=["POST", "GET"],
    allow_headers=["X-API-Key", "Content-Type"],
)

# ── Config — secrets MUST come from environment, no fallback defaults ─────────
try:
    GEMINI_API_KEY = os.environ["GEMINI_API_KEY"]
    API_KEY = os.environ["ATHLETIQ_API_KEY"]
except KeyError as e:
    raise RuntimeError(
        f"Missing required environment variable: {e}. "
        "Set GEMINI_API_KEY and ATHLETIQ_API_KEY before starting the app."
    )

GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Supabase is optional — if not configured, data logging is skipped (with a warning)
# rather than crashing the whole app, since training-data capture shouldn't take down
# the core analysis feature if misconfigured.
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY")
SUPABASE_MEDIA_BUCKET = "coaching-media"

supabase: Optional[Client] = None
if SUPABASE_URL and SUPABASE_SERVICE_KEY:
    supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
else:
    logging.getLogger("athletiq").warning(
        "SUPABASE_URL / SUPABASE_SERVICE_KEY not set — training data logging disabled"
    )


def log_interaction(
    endpoint: str,
    ai_response: str,
    processing_time_ms: int,
    sport: Optional[str] = None,
    question: Optional[str] = None,
    media_bytes: Optional[bytes] = None,
    media_type: Optional[str] = None,
):
    """Log an interaction for later coach review and training data export.
    Never raises — a failure here must not break the user-facing response."""
    if not supabase:
        return

    try:
        media_storage_path = None
        if media_bytes and media_type:
            ext = "jpg" if media_type == "image" else "mp4"
            date_prefix = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            filename = f"{uuid.uuid4()}.{ext}"
            media_storage_path = f"media/{date_prefix}/{filename}"

            supabase.storage.from_(SUPABASE_MEDIA_BUCKET).upload(
                media_storage_path,
                media_bytes,
                file_options={"content-type": f"{media_type}/{ext}"},
            )

        supabase.table("coaching_interactions").insert({
            "endpoint": endpoint,
            "sport": sport,
            "question": question,
            "media_storage_path": media_storage_path,
            "media_type": media_type,
            "ai_response": ai_response,
            "processing_time_ms": processing_time_ms,
        }).execute()

    except Exception as e:
        # Log and continue — never let data collection break the actual API response
        logging.getLogger("athletiq").error("Failed to log interaction: %s", e)

# ── Size limits ─────────────────────────────────────────────────────────────────
MAX_IMAGE_BYTES = 10 * 1024 * 1024    # 10 MB
MAX_VIDEO_BYTES = 100 * 1024 * 1024   # 100 MB
MAX_QUESTION_LEN = 1000
MAX_HISTORY_ITEMS = 20

# ── Auth ───────────────────────────────────────────────────────────────────────
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(key: str = Security(api_key_header)):
    if not key or key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return key


def estimate_b64_bytes(b64_string: str) -> int:
    """Approximate decoded byte size of a base64 string without fully decoding it."""
    padding = b64_string.count("=")
    return (len(b64_string) * 3) // 4 - padding


# ── Request / Response Models ─────────────────────────────────────────────────
class ImageAnalysisRequest(BaseModel):
    image_base64: str
    question: str = "Analyse this athlete's form and provide coaching feedback"
    sport: Optional[str] = None

    @field_validator("question")
    @classmethod
    def limit_question_len(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError(f"question must be under {MAX_QUESTION_LEN} characters")
        return v

    @field_validator("image_base64")
    @classmethod
    def limit_image_size(cls, v: str) -> str:
        if estimate_b64_bytes(v) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds maximum allowed size (10MB)")
        return v


class VideoAnalysisRequest(BaseModel):
    video_base64: str
    question: str = "Analyse this athlete's movement and provide coaching feedback"
    sport: Optional[str] = None
    max_frames: int = 6

    @field_validator("question")
    @classmethod
    def limit_question_len(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError(f"question must be under {MAX_QUESTION_LEN} characters")
        return v

    @field_validator("video_base64")
    @classmethod
    def limit_video_size(cls, v: str) -> str:
        if estimate_b64_bytes(v) > MAX_VIDEO_BYTES:
            raise ValueError("video exceeds maximum allowed size (100MB)")
        return v

    @field_validator("max_frames")
    @classmethod
    def clamp_frames(cls, v: int) -> int:
        return max(1, min(v, 10))


class ChatRequest(BaseModel):
    message: str
    history: Optional[list] = []
    sport: Optional[str] = None

    @field_validator("message")
    @classmethod
    def limit_message_len(cls, v: str) -> str:
        if len(v) > MAX_QUESTION_LEN:
            raise ValueError(f"message must be under {MAX_QUESTION_LEN} characters")
        return v

    @field_validator("history")
    @classmethod
    def limit_history(cls, v: list) -> list:
        if v and len(v) > MAX_HISTORY_ITEMS:
            raise ValueError(f"history limited to {MAX_HISTORY_ITEMS} items")
        return v


class AnalysisResponse(BaseModel):
    success: bool
    analysis: str
    frames_analysed: Optional[int] = None
    duration_seconds: Optional[float] = None
    processing_time_ms: int


class Flaw(BaseModel):
    label: str
    explanation: str
    x: int
    y: int
    severity: str


class AnnotatedAnalysisResponse(BaseModel):
    success: bool
    summary: str
    flaws: list[Flaw]
    recommended_exercises: list[str]
    annotated_image_base64: str
    processing_time_ms: int


class AnnotatedFrame(BaseModel):
    frame_number: int
    timestamp_seconds: float
    annotated_image_base64: str
    flaws: list[Flaw]


class AnnotatedVideoResponse(BaseModel):
    success: bool
    summary: str
    recommended_exercises: list[str]
    frames: list[AnnotatedFrame]
    duration_seconds: float
    processing_time_ms: int


# ── Gemini API Helper ──────────────────────────────────────────────────────────
def call_gemini(system_prompt: str, user_text: str, images_b64: Optional[list] = None, max_tokens: int = 600) -> str:
    """Call Gemini's generateContent endpoint. images_b64 is an optional list of
    base64-encoded JPEG strings (no data: prefix) to include alongside the text."""
    parts = []
    if images_b64:
        for img in images_b64:
            parts.append({"inline_data": {"mime_type": "image/jpeg", "data": img}})
    parts.append({"text": f"{system_prompt}\n\n{user_text}"})

    payload = json.dumps({
        "contents": [{"parts": parts}],
        "generationConfig": {"maxOutputTokens": max_tokens},
    }).encode()

    url = f"{GEMINI_URL_TEMPLATE.format(model=GEMINI_MODEL)}?key={GEMINI_API_KEY}"

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        return result["candidates"][0]["content"]["parts"][0]["text"].strip()
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            error_detail = json.loads(body).get("error", {}).get("message", "")
        except json.JSONDecodeError:
            error_detail = body.decode("utf-8", errors="replace")
        # Never log the URL itself — it contains the API key as a query param
        logger.error("Gemini API error %s: %s", e.code, error_detail)
        raise HTTPException(status_code=502, detail="Upstream model error — please try again")
    except urllib.error.URLError as e:
        logger.error("Gemini connection error: %s", e)
        raise HTTPException(status_code=503, detail="Model service temporarily unavailable")
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Gemini response shape: %s", e)
        raise HTTPException(status_code=502, detail="Unexpected response from model service")


# ── Frame Extraction ──────────────────────────────────────────────────────────
def extract_frames(video_path: str, max_frames: int = 6):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise HTTPException(status_code=400, detail="Could not open video file — invalid or corrupt format")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    duration = total / fps if fps > 0 else 0

    if total <= 0:
        cap.release()
        raise HTTPException(status_code=400, detail="Video contains no readable frames")

    indices = [int(total * i / max_frames) for i in range(max_frames)]
    frames_b64 = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            frames_b64.append(base64.b64encode(buf).decode())

    cap.release()
    return frames_b64, round(duration, 1)


# ── System Prompt ──────────────────────────────────────────────────────────────
def build_system_prompt(sport: Optional[str] = None) -> str:
    base = (
        "You are an expert sports performance analyst and coaching assistant. "
        "Analyse athlete movements for posture, biomechanics, technique, injury risks, "
        "and provide specific actionable improvements."
    )
    if sport:
        # sport is a free-text field from the client — keep it short and treat as plain text only
        safe_sport = sport[:50]
        base += f" You are specialising in {safe_sport} analysis."
    return base


# ══════════════════════════════════════════════════════════════════════════════
# ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"name": "AthletiQ API", "version": "1.0.0", "status": "running"}


@app.get("/health")
def health():
    return {"status": "healthy", "timestamp": time.time()}


# ── 1. Analyse Image ──────────────────────────────────────────────────────────
@app.post("/api/analyse-image", response_model=AnalysisResponse)
@limiter.limit("10/minute")
def analyse_image(request: Request, req: ImageAnalysisRequest, _=Depends(verify_api_key)):
    start = time.time()

    analysis = call_gemini(
        system_prompt=build_system_prompt(req.sport),
        user_text=req.question,
        images_b64=[req.image_base64],
    )
    ms = int((time.time() - start) * 1000)

    log_interaction(
        endpoint="analyse-image",
        ai_response=analysis,
        processing_time_ms=ms,
        sport=req.sport,
        question=req.question,
        media_bytes=base64.b64decode(req.image_base64),
        media_type="image",
    )

    return AnalysisResponse(success=True, analysis=analysis, processing_time_ms=ms)


# ── 2. Analyse Video ──────────────────────────────────────────────────────────
@app.post("/api/analyse-video", response_model=AnalysisResponse)
@limiter.limit("5/minute")
def analyse_video(request: Request, req: VideoAnalysisRequest, _=Depends(verify_api_key)):
    start = time.time()
    max_frames = req.max_frames  # already clamped by validator

    video_data = base64.b64decode(req.video_base64)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_data)
            tmp_path = f.name

        frames, duration = extract_frames(tmp_path, max_frames)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video")

    system = build_system_prompt(req.sport)
    frame_analyses = []
    for i, frame in enumerate(frames):
        frame_question = f"Frame {i + 1} of {len(frames)} (video duration: {duration}s): {req.question}"
        analysis = call_gemini(
            system_prompt=system,
            user_text=frame_question,
            images_b64=[frame],
            max_tokens=300,
        )
        frame_analyses.append(f"Frame {i + 1}: {analysis}")

    summary_system = (
        "You are an expert sports performance analyst. Summarise the following frame-by-frame "
        "analysis into one cohesive coaching report with: 1) Key observations 2) Main issues "
        "identified 3) Specific actionable improvements 4) Injury risk assessment."
    )
    summary_text = (
        f"Sport: {req.sport or 'general'}\nQuestion: {req.question}\nVideo duration: {duration}s\n\n"
        + "\n\n".join(frame_analyses)
    )
    final_analysis = call_gemini(system_prompt=summary_system, user_text=summary_text, max_tokens=800)
    ms = int((time.time() - start) * 1000)

    log_interaction(
        endpoint="analyse-video",
        ai_response=final_analysis,
        processing_time_ms=ms,
        sport=req.sport,
        question=req.question,
        media_bytes=video_data,
        media_type="video",
    )

    return AnalysisResponse(
        success=True,
        analysis=final_analysis,
        frames_analysed=len(frames),
        duration_seconds=duration,
        processing_time_ms=ms,
    )


# ── 3. Chat ────────────────────────────────────────────────────────────────────
@app.post("/api/chat", response_model=AnalysisResponse)
@limiter.limit("20/minute")
def chat(request: Request, req: ChatRequest, _=Depends(verify_api_key)):
    start = time.time()

    history_text = ""
    for h in req.history:
        if isinstance(h, dict) and "role" in h and "content" in h and h["role"] in ("user", "assistant"):
            history_text += f"{h['role']}: {str(h['content'])[:MAX_QUESTION_LEN]}\n"

    user_text = req.message
    if history_text:
        user_text = f"Previous conversation:\n{history_text}\nNew question: {req.message}"

    analysis = call_gemini(system_prompt=build_system_prompt(req.sport), user_text=user_text)
    ms = int((time.time() - start) * 1000)

    return AnalysisResponse(success=True, analysis=analysis, processing_time_ms=ms)


# ── 4. Analyse Image with Visual Annotations ──────────────────────────────────
@app.post("/api/analyse-image-annotated", response_model=AnnotatedAnalysisResponse)
@limiter.limit("10/minute")
def analyse_image_annotated(request: Request, req: ImageAnalysisRequest, _=Depends(verify_api_key)):
    start = time.time()

    image_bytes = base64.b64decode(req.image_base64)

    try:
        width, height = get_image_dimensions(image_bytes)
    except ValueError:
        raise HTTPException(status_code=400, detail="Could not read image dimensions")

    try:
        result = call_gemini_with_annotations(
            image_b64=req.image_base64,
            image_width=width,
            image_height=height,
            question=req.question,
            api_key=GEMINI_API_KEY,
            sport=req.sport,
        )
    except RuntimeError as e:
        logger.error("Gemini annotation error: %s", e)
        raise HTTPException(status_code=502, detail="Upstream model error — please try again")

    flaws = result.get("flaws", [])

    try:
        annotated_bytes = draw_annotations(image_bytes, flaws)
        annotated_b64 = base64.b64encode(annotated_bytes).decode()
    except ValueError as e:
        logger.error("Annotation drawing error: %s", e)
        annotated_b64 = req.image_base64

    ms = int((time.time() - start) * 1000)

    log_interaction(
        endpoint="analyse-image-annotated",
        ai_response=result.get("summary", ""),
        processing_time_ms=ms,
        sport=req.sport,
        question=req.question,
        media_bytes=image_bytes,
        media_type="image",
    )

    return AnnotatedAnalysisResponse(
        success=True,
        summary=result.get("summary", ""),
        flaws=flaws,
        recommended_exercises=result.get("recommended_exercises", []),
        annotated_image_base64=annotated_b64,
        processing_time_ms=ms,
    )


# ── 5. Analyse Video with Visual Annotations (key frames only) ───────────────
@app.post("/api/analyse-video-annotated", response_model=AnnotatedVideoResponse)
@limiter.limit("5/minute")
def analyse_video_annotated(request: Request, req: VideoAnalysisRequest, _=Depends(verify_api_key)):
    start = time.time()
    max_frames = req.max_frames

    video_data = base64.b64decode(req.video_base64)

    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(video_data)
            tmp_path = f.name

        raw_frames, duration = extract_frames(tmp_path, max_frames)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

    if not raw_frames:
        raise HTTPException(status_code=400, detail="Could not extract frames from video")

    annotated_frames = []
    all_flaws_for_summary = []
    last_width, last_height = None, None

    for i, frame_b64 in enumerate(raw_frames):
        frame_bytes = base64.b64decode(frame_b64)

        try:
            width, height = get_image_dimensions(frame_bytes)
            last_width, last_height = width, height
        except ValueError:
            continue

        frame_question = (
            f"{req.question} This is frame {i + 1} of {len(raw_frames)} from a video "
            f"(video duration: {duration}s)."
        )

        try:
            result = call_gemini_with_annotations(
                image_b64=frame_b64,
                image_width=width,
                image_height=height,
                question=frame_question,
                api_key=GEMINI_API_KEY,
                sport=req.sport,
            )
        except RuntimeError as e:
            logger.error("Gemini annotation error on frame %d: %s", i, e)
            continue

        flaws = result.get("flaws", [])
        if not flaws:
            continue

        try:
            annotated_bytes = draw_annotations(frame_bytes, flaws)
            annotated_b64 = base64.b64encode(annotated_bytes).decode()
        except ValueError:
            annotated_b64 = frame_b64

        timestamp = round((i / max(len(raw_frames) - 1, 1)) * duration, 1)

        annotated_frames.append(AnnotatedFrame(
            frame_number=i + 1,
            timestamp_seconds=timestamp,
            annotated_image_base64=annotated_b64,
            flaws=flaws,
        ))
        all_flaws_for_summary.extend(flaws)

    if annotated_frames and last_width and last_height:
        summary_question = (
            "Summarise these biomechanical issues found across a video into one cohesive "
            "coaching summary, plus a list of recommended exercises. "
            f"Sport: {req.sport or 'general'}.\n\n"
            + "\n".join(f"- {f.get('label')}: {f.get('explanation')}" for f in all_flaws_for_summary)
        )
        try:
            summary_result = call_gemini_with_annotations(
                image_b64=raw_frames[0],
                image_width=last_width,
                image_height=last_height,
                question=summary_question,
                api_key=GEMINI_API_KEY,
                sport=req.sport,
            )
            overall_summary = summary_result.get("summary", "")
            recommended_exercises = summary_result.get("recommended_exercises", [])
        except RuntimeError:
            overall_summary = f"Found {len(annotated_frames)} frame(s) with notable form issues."
            recommended_exercises = []
    else:
        overall_summary = "No significant form issues detected in this video."
        recommended_exercises = []

    ms = int((time.time() - start) * 1000)

    log_interaction(
        endpoint="analyse-video-annotated",
        ai_response=overall_summary,
        processing_time_ms=ms,
        sport=req.sport,
        question=req.question,
        media_bytes=video_data,
        media_type="video",
    )

    return AnnotatedVideoResponse(
        success=True,
        summary=overall_summary,
        recommended_exercises=recommended_exercises,
        frames=annotated_frames,
        duration_seconds=duration,
        processing_time_ms=ms,
    )
