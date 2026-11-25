"""
Utility functions for classroom talking detection.
Handles MAR calculation, drawing, validation, and safe I/O.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mouth Aspect Ratio (MAR)
# ---------------------------------------------------------------------------

# MediaPipe Face Mesh landmark indices for inner mouth
# Upper lip: 13, lower lip: 14, left corner: 78, right corner: 308
UPPER_LIP_IDX = 13
LOWER_LIP_IDX = 14
LEFT_MOUTH_IDX = 78
RIGHT_MOUTH_IDX = 308

# Alternative more robust set (outer + inner)
UPPER_INNER = 13
LOWER_INNER = 14
LEFT_CORNER = 61
RIGHT_CORNER = 291


def calculate_mar(landmarks: np.ndarray, image_shape: Tuple[int, int]) -> float:
    """
    Calculate Mouth Aspect Ratio from MediaPipe Face Mesh landmarks.

    MAR = vertical distance between upper & lower inner lip
          / horizontal distance between mouth corners

    Args:
        landmarks: (468, 3) or (478, 3) array of normalized landmarks
        image_shape: (height, width) of the source image

    Returns:
        MAR value (float). Returns 0.0 if landmarks are invalid.
    """
    if landmarks is None or len(landmarks) < 310:
        return 0.0

    h, w = image_shape[:2]

    try:
        # Convert normalized landmarks to pixel coordinates
        upper = landmarks[UPPER_INNER]
        lower = landmarks[LOWER_INNER]
        left = landmarks[LEFT_CORNER]
        right = landmarks[RIGHT_CORNER]

        upper_px = np.array([upper[0] * w, upper[1] * h])
        lower_px = np.array([lower[0] * w, lower[1] * h])
        left_px = np.array([left[0] * w, left[1] * h])
        right_px = np.array([right[0] * w, right[1] * h])

        vertical = np.linalg.norm(upper_px - lower_px)
        horizontal = np.linalg.norm(left_px - right_px)

        if horizontal < 1e-6:
            return 0.0

        mar = float(vertical / horizontal)
        return max(0.0, min(mar, 2.0))  # clamp to sensible range
    except Exception as e:
        logger.debug("MAR calculation failed: %s", e)
        return 0.0


def is_face_visible(landmarks: Optional[np.ndarray], min_landmarks: int = 100) -> bool:
    """Return True if enough landmarks are present to compute MAR."""
    if landmarks is None:
        return False
    return len(landmarks) >= min_landmarks


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

COLOR_QUIET = (0, 200, 0)          # Green
COLOR_TALKING = (0, 0, 255)        # Red
COLOR_FACE_NOT_VISIBLE = (0, 165, 255)  # Orange
COLOR_ID_BG = (30, 30, 30)


def draw_student_overlay(
    frame: np.ndarray,
    bbox: Tuple[int, int, int, int],
    student_id: int,
    status: str,
    mar: float,
    talking_duration: float = 0.0,
    confidence: float = 0.0,
) -> np.ndarray:
    """
    Draw bounding box, ID, status, MAR and duration on the frame.

    status: "TALKING" | "QUIET" | "Face Not Visible" | "LIKELY TALKING"
    """
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = frame.shape[:2]

    # Clamp box to frame
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w - 1, x2), min(h - 1, y2)

    if status == "TALKING" or status == "LIKELY TALKING":
        color = COLOR_TALKING
    elif status == "Face Not Visible":
        color = COLOR_FACE_NOT_VISIBLE
    else:
        color = COLOR_QUIET

    thickness = 3 if status in ("TALKING", "LIKELY TALKING") else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # Label background
    label_lines = [
        f"Student #{student_id}",
        f"{status}",
        f"MAR: {mar:.3f}",
    ]
    if talking_duration > 0:
        label_lines.append(f"Talk: {talking_duration:.1f}s")

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 1
    line_height = 18
    padding = 4

    max_text_w = 0
    for line in label_lines:
        (tw, _), _ = cv2.getTextSize(line, font, font_scale, font_thickness)
        max_text_w = max(max_text_w, tw)

    box_h = len(label_lines) * line_height + padding * 2
    box_w = max_text_w + padding * 2

    # Prefer above the box; fall back below if near top
    label_y1 = y1 - box_h - 4
    if label_y1 < 0:
        label_y1 = y2 + 4
    label_y2 = label_y1 + box_h
    label_x1 = x1
    label_x2 = x1 + box_w

    # Background
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), COLOR_ID_BG, -1)
    cv2.rectangle(frame, (label_x1, label_y1), (label_x2, label_y2), color, 1)

    for i, line in enumerate(label_lines):
        ty = label_y1 + padding + (i + 1) * line_height - 4
        cv2.putText(
            frame,
            line,
            (label_x1 + padding, ty),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA,
        )

    return frame


def draw_dashboard_overlay(
    frame: np.ndarray,
    total_students: int,
    talking_count: int,
    quiet_count: int,
) -> np.ndarray:
    """Draw a small live stats panel in the top-left corner."""
    h, w = frame.shape[:2]
    panel_w, panel_h = 220, 90
    x1, y1 = 10, 10
    x2, y2 = x1 + panel_w, y1 + panel_h

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Students: {total_students}", (x1 + 10, y1 + 25),
                font, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Talking:  {talking_count}", (x1 + 10, y1 + 50),
                font, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Quiet:    {quiet_count}", (x1 + 10, y1 + 75),
                font, 0.55, (0, 200, 0), 1, cv2.LINE_AA)

    return frame


# ---------------------------------------------------------------------------
# Input validation & helpers
# ---------------------------------------------------------------------------

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SUPPORTED_VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def is_valid_image(path_or_bytes: Any) -> bool:
    """Quick check that data can be opened as an image."""
    try:
        if isinstance(path_or_bytes, (str, Path)):
            img = cv2.imread(str(path_or_bytes))
            return img is not None and img.size > 0
        elif isinstance(path_or_bytes, bytes):
            arr = np.frombuffer(path_or_bytes, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img is not None and img.size > 0
        elif isinstance(path_or_bytes, np.ndarray):
            return path_or_bytes.size > 0 and len(path_or_bytes.shape) >= 2
        return False
    except Exception:
        return False


def load_image_safely(file) -> Optional[np.ndarray]:
    """Load uploaded Streamlit file or path into BGR numpy array."""
    try:
        if hasattr(file, "read"):
            data = file.read()
            if not data:
                return None
            arr = np.frombuffer(data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            return img if img is not None and img.size > 0 else None
        else:
            img = cv2.imread(str(file))
            return img if img is not None and img.size > 0 else None
    except Exception as e:
        logger.warning("Failed to load image: %s", e)
        return None


def get_video_info(cap: cv2.VideoCapture) -> Dict[str, Any]:
    """Extract basic metadata from an open VideoCapture."""
    return {
        "fps": cap.get(cv2.CAP_PROP_FPS) or 25.0,
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
        "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
    }


def safe_release(*caps_or_writers) -> None:
    """Release any number of VideoCapture / VideoWriter objects."""
    for obj in caps_or_writers:
        if obj is None:
            continue
        try:
            if hasattr(obj, "release"):
                obj.release()
        except Exception as e:
            logger.debug("Release warning: %s", e)


def bbox_iou(box_a: Tuple[float, float, float, float],
             box_b: Tuple[float, float, float, float]) -> float:
    """Compute IoU between two (x1, y1, x2, y2) boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0

    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def center_distance(box_a: Tuple[float, ...], box_b: Tuple[float, ...]) -> float:
    """Euclidean distance between box centers."""
    ca = ((box_a[0] + box_a[2]) / 2, (box_a[1] + box_a[3]) / 2)
    cb = ((box_b[0] + box_b[2]) / 2, (box_b[1] + box_b[3]) / 2)
    return float(np.hypot(ca[0] - cb[0], ca[1] - cb[1]))
