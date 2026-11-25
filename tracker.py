"""
Core tracking + talking detection engine.

Combines Ultralytics YOLO (with ByteTrack) for multi-student tracking
and MediaPipe Face Mesh for Mouth Aspect Ratio (MAR) analysis.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

from utils import (
    calculate_mar,
    draw_dashboard_overlay,
    draw_student_overlay,
    is_face_visible,
    bbox_iou,
    center_distance,
)

logger = logging.getLogger(__name__)


@dataclass
class StudentState:
    """Persistent state for one tracked student."""
    track_id: int
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    mar: float = 0.0
    status: str = "QUIET"                 # QUIET | TALKING | Face Not Visible | LIKELY TALKING
    consecutive_open: int = 0             # frames with MAR > threshold
    talking_frames: int = 0               # total frames marked talking
    talking_duration: float = 0.0         # seconds
    talking_events: int = 0               # number of distinct talking segments
    last_was_talking: bool = False
    confidence: float = 0.0
    last_seen: float = field(default_factory=time.time)
    face_visible: bool = False


class ClassroomTalkTracker:
    """
    Production tracker that:
    1. Detects & tracks every person with YOLO + ByteTrack
    2. Associates MediaPipe Face Mesh results to each track
    3. Computes MAR and applies temporal smoothing
    4. Maintains per-student talking statistics
    """

    def __init__(
        self,
        yolo_weights: str = "yolo11n.pt",
        conf_threshold: float = 0.40,
        mar_threshold: float = 0.35,
        consecutive_frames: int = 6,
        iou_match_threshold: float = 0.25,
        max_age_seconds: float = 3.0,
        device: Optional[str] = None,
    ):
        self.conf_threshold = conf_threshold
        self.mar_threshold = mar_threshold
        self.consecutive_frames = max(1, consecutive_frames)
        self.iou_match_threshold = iou_match_threshold
        self.max_age_seconds = max_age_seconds

        # ---- YOLO ----
        try:
            self.model = YOLO(yolo_weights)
            # Force person-only class (COCO class 0)
            self.model.overrides["classes"] = [0]
        except Exception as e:
            logger.error("Failed to load YOLO weights %s: %s", yolo_weights, e)
            raise RuntimeError(
                f"Could not load YOLO model '{yolo_weights}'. "
                "It will be downloaded automatically on first run if internet is available."
            ) from e

        # Auto device
        if device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        logger.info("YOLO running on device: %s", self.device)

        # ---- MediaPipe Face Mesh ----
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=30,               # classroom can have many students
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )
        # Separate instance for single-image mode (static)
        self.face_mesh_static = self.mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=30,
            refine_landmarks=True,
            min_detection_confidence=0.4,
            min_tracking_confidence=0.4,
        )

        # ---- State ----
        self.students: Dict[int, StudentState] = {}
        self.frame_count = 0
        self.fps = 25.0                     # updated externally for video
        self._is_image_mode = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all tracking state (call between videos / sessions)."""
        self.students.clear()
        self.frame_count = 0

    def update_params(
        self,
        conf_threshold: Optional[float] = None,
        mar_threshold: Optional[float] = None,
        consecutive_frames: Optional[int] = None,
    ) -> None:
        if conf_threshold is not None:
            self.conf_threshold = conf_threshold
        if mar_threshold is not None:
            self.mar_threshold = mar_threshold
        if consecutive_frames is not None:
            self.consecutive_frames = max(1, consecutive_frames)

    def process_frame(
        self,
        frame: np.ndarray,
        is_image: bool = False,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Process a single BGR frame.

        Returns:
            annotated_frame, stats_dict
        """
        if frame is None or frame.size == 0:
            return frame, self._empty_stats()

        self._is_image_mode = is_image
        self.frame_count += 1
        h, w = frame.shape[:2]
        now = time.time()

        # 1. YOLO tracking (persons only)
        tracks = self._run_yolo(frame)

        # 2. MediaPipe Face Mesh
        face_results = self._run_face_mesh(frame, static=is_image)

        # 3. Associate faces → tracks
        face_assignments = self._associate_faces_to_tracks(tracks, face_results, (h, w))

        # 4. Update student states
        active_ids = set()
        for track in tracks:
            tid = track["id"]
            active_ids.add(tid)
            bbox = track["bbox"]
            conf = track["conf"]

            if tid not in self.students:
                self.students[tid] = StudentState(track_id=tid)

            st = self.students[tid]
            st.bbox = bbox
            st.confidence = conf
            st.last_seen = now

            # Face / MAR
            landmarks = face_assignments.get(tid)
            if landmarks is not None and is_face_visible(landmarks):
                st.face_visible = True
                st.mar = calculate_mar(landmarks, (h, w))
            else:
                st.face_visible = False
                st.mar = 0.0

            # Status logic
            self._update_talking_status(st, is_image)

        # 5. Age-out disappeared tracks
        to_remove = [
            tid for tid, st in self.students.items()
            if (now - st.last_seen) > self.max_age_seconds and tid not in active_ids
        ]
        for tid in to_remove:
            del self.students[tid]

        # 6. Draw
        annotated = frame.copy()
        talking_count = 0
        quiet_count = 0

        for tid, st in sorted(self.students.items()):
            if tid not in active_ids:
                continue
            annotated = draw_student_overlay(
                annotated,
                st.bbox,
                st.track_id,
                st.status,
                st.mar,
                st.talking_duration,
                st.confidence,
            )
            if st.status in ("TALKING", "LIKELY TALKING"):
                talking_count += 1
            elif st.status == "QUIET":
                quiet_count += 1

        total = talking_count + quiet_count + sum(
            1 for tid, st in self.students.items()
            if tid in active_ids and st.status == "Face Not Visible"
        )
        annotated = draw_dashboard_overlay(annotated, total, talking_count, quiet_count)

        stats = {
            "total_students": total,
            "talking": talking_count,
            "quiet": quiet_count,
            "face_not_visible": total - talking_count - quiet_count,
            "students": {
                tid: {
                    "status": st.status,
                    "mar": round(st.mar, 4),
                    "talking_duration": round(st.talking_duration, 2),
                    "talking_events": st.talking_events,
                    "face_visible": st.face_visible,
                }
                for tid, st in self.students.items()
                if tid in active_ids
            },
        }
        return annotated, stats

    def get_leaderboard(self) -> List[Dict[str, Any]]:
        """Return ranked list of students by total talking duration."""
        rows = []
        for tid, st in self.students.items():
            rows.append({
                "Student ID": f"Student #{tid}",
                "Talking Events": st.talking_events,
                "Total Duration (s)": round(st.talking_duration, 2),
                "Current Status": st.status,
                "Last MAR": round(st.mar, 4),
            })
        rows.sort(key=lambda r: r["Total Duration (s)"], reverse=True)
        return rows

    def close(self) -> None:
        """Release MediaPipe resources."""
        try:
            self.face_mesh.close()
            self.face_mesh_static.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_yolo(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """Run YOLO track and return list of {id, bbox, conf}."""
        try:
            results = self.model.track(
                frame,
                persist=True,
                conf=self.conf_threshold,
                classes=[0],          # person only
                verbose=False,
                device=self.device,
            )
        except Exception as e:
            logger.warning("YOLO track failed: %s", e)
            return []

        tracks = []
        if not results or results[0].boxes is None:
            return tracks

        boxes = results[0].boxes
        if boxes.id is None:
            # No tracking IDs yet (first frames) – assign temporary sequential IDs
            for i, box in enumerate(boxes):
                xyxy = box.xyxy.cpu().numpy().flatten()
                conf = float(box.conf.cpu().numpy())
                tracks.append({
                    "id": i + 1,
                    "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                    "conf": conf,
                })
            return tracks

        ids = boxes.id.cpu().numpy().astype(int)
        xyxys = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()

        for tid, xyxy, conf in zip(ids, xyxys, confs):
            tracks.append({
                "id": int(tid),
                "bbox": (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                "conf": float(conf),
            })
        return tracks

    def _run_face_mesh(
        self, frame: np.ndarray, static: bool = False
    ) -> List[np.ndarray]:
        """Return list of landmark arrays (each N×3)."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mesh = self.face_mesh_static if static else self.face_mesh
            results = mesh.process(rgb)
            if not results.multi_face_landmarks:
                return []
            faces = []
            for face_lms in results.multi_face_landmarks:
                lms = np.array(
                    [[lm.x, lm.y, lm.z] for lm in face_lms.landmark],
                    dtype=np.float32,
                )
                faces.append(lms)
            return faces
        except Exception as e:
            logger.debug("Face Mesh failed: %s", e)
            return []

    def _associate_faces_to_tracks(
        self,
        tracks: List[Dict],
        faces: List[np.ndarray],
        image_shape: Tuple[int, int],
    ) -> Dict[int, np.ndarray]:
        """
        Greedy match each face to the best overlapping track by IoU of
        face bounding box vs student bounding box.
        """
        if not tracks or not faces:
            return {}

        h, w = image_shape
        assignments: Dict[int, np.ndarray] = {}
        used_faces = set()

        # Pre-compute face bboxes (from landmarks)
        face_boxes = []
        for face in faces:
            xs = face[:, 0] * w
            ys = face[:, 1] * h
            face_boxes.append(
                (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
            )

        # Sort tracks by confidence descending for better assignment
        sorted_tracks = sorted(tracks, key=lambda t: t["conf"], reverse=True)

        for track in sorted_tracks:
            best_iou = 0.0
            best_idx = -1
            tbox = track["bbox"]

            for fi, fbox in enumerate(face_boxes):
                if fi in used_faces:
                    continue
                iou = bbox_iou(tbox, fbox)
                # Also consider center distance as secondary signal
                if iou > best_iou:
                    best_iou = iou
                    best_idx = fi

            if best_idx >= 0 and best_iou >= self.iou_match_threshold:
                assignments[track["id"]] = faces[best_idx]
                used_faces.add(best_idx)
            else:
                # Fallback: nearest center if reasonably close
                tcx = (tbox[0] + tbox[2]) / 2
                tcy = (tbox[1] + tbox[3]) / 2
                best_dist = 1e9
                best_idx = -1
                for fi, fbox in enumerate(face_boxes):
                    if fi in used_faces:
                        continue
                    fcx = (fbox[0] + fbox[2]) / 2
                    fcy = (fbox[1] + fbox[3]) / 2
                    dist = np.hypot(tcx - fcx, tcy - fcy)
                    # Accept if face center is inside the person box or very close
                    if (tbox[0] <= fcx <= tbox[2] and tbox[1] <= fcy <= tbox[3]) or dist < 80:
                        if dist < best_dist:
                            best_dist = dist
                            best_idx = fi
                if best_idx >= 0:
                    assignments[track["id"]] = faces[best_idx]
                    used_faces.add(best_idx)

        return assignments

    def _update_talking_status(self, st: StudentState, is_image: bool) -> None:
        """Apply MAR threshold + temporal smoothing."""
        if not st.face_visible:
            st.status = "Face Not Visible"
            st.consecutive_open = 0
            # Do not reset talking_duration – keep historical stats
            return

        is_open = st.mar >= self.mar_threshold

        if is_image:
            # Single-frame decision
            st.status = "LIKELY TALKING" if is_open else "QUIET"
            if is_open:
                st.talking_frames += 1
            return

        # Video / live mode – temporal smoothing
        if is_open:
            st.consecutive_open += 1
        else:
            st.consecutive_open = 0

        was_talking = st.last_was_talking
        is_talking = st.consecutive_open >= self.consecutive_frames

        if is_talking:
            st.status = "TALKING"
            st.talking_frames += 1
            # Accumulate duration using current FPS estimate
            st.talking_duration += 1.0 / max(self.fps, 1.0)
            if not was_talking:
                st.talking_events += 1
            st.last_was_talking = True
        else:
            st.status = "QUIET"
            st.last_was_talking = False

    def _empty_stats(self) -> Dict[str, Any]:
        return {
            "total_students": 0,
            "talking": 0,
            "quiet": 0,
            "face_not_visible": 0,
            "students": {},
        }
