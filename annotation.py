import base64
import json
import urllib.request
import urllib.error
from typing import Optional

import cv2
import numpy as np


GEMINI_ANNOTATION_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary": {
            "type": "STRING",
            "description": "Overall coaching summary in 2-4 sentences",
        },
        "flaws": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "label": {
                        "type": "STRING",
                        "description": "Short name of the issue, e.g. 'Knee valgus'",
                    },
                    "explanation": {
                        "type": "STRING",
                        "description": "1-2 sentence explanation of why this is a problem and the injury risk",
                    },
                    "x": {
                        "type": "INTEGER",
                        "description": "X pixel coordinate of the issue location in the image",
                    },
                    "y": {
                        "type": "INTEGER",
                        "description": "Y pixel coordinate of the issue location in the image",
                    },
                    "severity": {
                        "type": "STRING",
                        "enum": ["low", "medium", "high"],
                        "description": "How urgent this issue is",
                    },
                },
                "required": ["label", "explanation", "x", "y", "severity"],
            },
        },
        "recommended_exercises": {
            "type": "ARRAY",
            "items": {"type": "STRING"},
            "description": "Specific exercises to address the flaws found",
        },
    },
    "required": ["summary", "flaws", "recommended_exercises"],
}


def call_gemini_with_annotations(
    image_b64: str,
    image_width: int,
    image_height: int,
    question: str,
    api_key: str,
    sport: Optional[str] = None,
) -> dict:
    """Call Gemini asking for structured flaw data with pixel coordinates,
    so the server can draw markers directly on the image."""

    sport_line = f" The athlete is performing: {sport}." if sport else ""

    prompt_text = (
        "You are an expert sports performance analyst and physiotherapist. "
        f"Analyse this athlete's image (dimensions: {image_width}x{image_height} pixels)."
        f"{sport_line} {question}\n\n"
        "For each issue you identify, provide the approximate pixel x,y coordinate "
        "of where that issue is visible in the image (e.g. the knee joint, the shoulder, "
        "the lower back). Coordinates must be within the image bounds: "
        f"x between 0 and {image_width}, y between 0 and {image_height}. "
        "Only include genuine biomechanical or postural issues — do not invent flaws "
        "if the form looks correct; in that case return an empty flaws array."
    )

    payload = json.dumps({
        "contents": [{
            "parts": [
                {"inline_data": {"mime_type": "image/jpeg", "data": image_b64}},
                {"text": prompt_text},
            ]
        }],
        "generationConfig": {
            "responseMimeType": "application/json",
            "responseSchema": GEMINI_ANNOTATION_SCHEMA,
        },
    }).encode()

    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"gemini-2.5-flash:generateContent?key={api_key}"
    )

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            result = json.loads(r.read())
        raw_text = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(raw_text)
    except urllib.error.HTTPError as e:
        error_body = e.read().decode(errors="replace")
        raise RuntimeError(f"Gemini API error {e.code}: {error_body}")
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        raise RuntimeError(f"Unexpected Gemini response shape: {e}")


def draw_annotations(image_bytes: bytes, flaws: list) -> bytes:
    """Draw circles + labels on the image at each flaw's coordinates.
    Returns JPEG-encoded bytes of the annotated image."""

    severity_colors = {
        "high": (40, 40, 230),     # red (BGR)
        "medium": (30, 160, 230),  # amber
        "low": (60, 200, 60),      # green
    }

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image for annotation")

    height, width = img.shape[:2]

    for i, flaw in enumerate(flaws):
        x = max(0, min(int(flaw.get("x", 0)), width - 1))
        y = max(0, min(int(flaw.get("y", 0)), height - 1))
        severity = flaw.get("severity", "medium")
        color = severity_colors.get(severity, severity_colors["medium"])
        label = flaw.get("label", f"Issue {i+1}")

        # Outer ring + inner dot marker
        cv2.circle(img, (x, y), 18, color, 2)
        cv2.circle(img, (x, y), 4, color, -1)

        # Label background for readability
        label_text = f"{i+1}. {label}"
        (text_w, text_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        label_x = min(x + 24, width - text_w - 8)
        label_y = max(y - 10, text_h + 8)

        cv2.rectangle(
            img,
            (label_x - 4, label_y - text_h - 4),
            (label_x + text_w + 4, label_y + 4),
            (20, 20, 20),
            -1,
        )
        cv2.putText(
            img, label_text, (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA,
        )

    success, buffer = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not success:
        raise ValueError("Could not encode annotated image")

    return buffer.tobytes()


def get_image_dimensions(image_bytes: bytes) -> tuple:
    """Returns (width, height) of an image from its raw bytes."""
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    height, width = img.shape[:2]
    return width, height
