import base64
import json
import logging
import os
import tempfile
import time
import urllib.error
import urllib.request
from typing import Optional

import cv2
from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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
    GROQ_API_KEY = os.environ["GROQ_API_KEY"]
    API_KEY = os.environ["ATHLETIQ_API_KEY"]
except KeyError as e:
    raise RuntimeError(
        f"Missing required environment variable: {e}. "
        "Set GROQ_API_KEY and ATHLETIQ_API_KEY before starting the app."
    )

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


# ── Groq API Helper ───────────────────────────────────────────────────────────
def call_groq(messages: list, model: str = "llama-3.3-70b-versatile", max_tokens: int = 600) -> str:
    payload = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()

    req = urllib.request.Request(
        "https://api.groq.com/openai/v1/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        return result["choices"][0]["message"]["content"].strip()
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            error_detail = json.loads(body).get("error", {}).get("message", "")
        except json.JSONDecodeError:
            error_detail = ""
        logger.error("Groq API error %s: %s", e.code, error_detail)
        # Don't leak upstream error internals to the client
        raise HTTPException(status_code=502, detail="Upstream model error — please try again")
    except urllib.error.URLError as e:
        logger.error("Groq connection error: %s", e)
        raise HTTPException(status_code=503, detail="Model service temporarily unavailable")
    except (KeyError, IndexError) as e:
        logger.error("Unexpected Groq response shape: %s", e)
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

    messages = [
        {"role": "system", "content": build_system_prompt(req.sport)},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{req.image_base64}"}},
            {"type": "text", "text": req.question},
        ]},
    ]

    analysis = call_groq(messages, model="meta-llama/llama-4-scout-17b-16e-instruct")
    ms = int((time.time() - start) * 1000)

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
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{frame}"}},
                {"type": "text", "text": f"Frame {i + 1} of {len(frames)} (video duration: {duration}s): {req.question}"},
            ]},
        ]
        analysis = call_groq(messages, model="meta-llama/llama-4-scout-17b-16e-instruct", max_tokens=300)
        frame_analyses.append(f"Frame {i + 1}: {analysis}")

    summary_messages = [
        {"role": "system", "content": (
            "You are an expert sports performance analyst. Summarise the following frame-by-frame "
            "analysis into one cohesive coaching report with: 1) Key observations 2) Main issues "
            "identified 3) Specific actionable improvements 4) Injury risk assessment."
        )},
        {"role": "user", "content": (
            f"Sport: {req.sport or 'general'}\nQuestion: {req.question}\nVideo duration: {duration}s\n\n"
            + "\n\n".join(frame_analyses)
        )},
    ]
    final_analysis = call_groq(summary_messages, max_tokens=800)
    ms = int((time.time() - start) * 1000)

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

    messages = [{"role": "system", "content": build_system_prompt(req.sport)}]
    for h in req.history:
        if isinstance(h, dict) and "role" in h and "content" in h and h["role"] in ("user", "assistant"):
            messages.append({"role": h["role"], "content": str(h["content"])[:MAX_QUESTION_LEN]})
    messages.append({"role": "user", "content": req.message})

    analysis = call_groq(messages)
    ms = int((time.time() - start) * 1000)

    return AnalysisResponse(success=True, analysis=analysis, processing_time_ms=ms)
