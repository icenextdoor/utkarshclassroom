"""
Classroom Talking Detection – Streamlit Web Application

Production-ready multi-modal interface supporting:
  • Image / batch image upload
  • Video upload + download of annotated result
  • Live webcam / RTSP stream
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from tracker import ClassroomTalkTracker
from utils import load_image_safely, safe_release, get_video_info

# ---------------------------------------------------------------------------
# Logging & page config
# ---------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Classroom Talking Detector",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Model caching (prevents reloads & memory leaks)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models (YOLO + MediaPipe)…")
def load_tracker(
    conf: float = 0.40,
    mar: float = 0.35,
    consec: int = 6,
) -> ClassroomTalkTracker:
    """
    Cached resource: loads YOLO weights (auto-downloads if missing)
    and MediaPipe Face Mesh once per session.
    """
    # Prefer yolo11n; fall back to yolov8n if needed
    weights = "yolo11n.pt"
    try:
        tracker = ClassroomTalkTracker(
            yolo_weights=weights,
            conf_threshold=conf,
            mar_threshold=mar,
            consecutive_frames=consec,
        )
    except Exception:
        logger.warning("yolo11n.pt failed, falling back to yolov8n.pt")
        tracker = ClassroomTalkTracker(
            yolo_weights="yolov8n.pt",
            conf_threshold=conf,
            mar_threshold=mar,
            consecutive_frames=consec,
        )
    return tracker


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

def render_sidebar() -> Dict[str, Any]:
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")

    mode = st.sidebar.radio(
        "Input Mode",
        ["🖼️ Image(s)", "🎬 Video", "📷 Live Camera / RTSP"],
        index=0,
    )

    st.sidebar.markdown("### Detection Parameters")

    mar_thresh = st.sidebar.slider(
        "MAR Sensitivity (Mouth Open Threshold)",
        min_value=0.15,
        max_value=0.75,
        value=0.35,
        step=0.01,
        help="Higher = less sensitive (only wide-open mouths). "
             "Lower = more sensitive (detects subtle talking).",
    )

    consec = st.sidebar.slider(
        "Consecutive Frames Required",
        min_value=1,
        max_value=30,
        value=6,
        step=1,
        help="Number of consecutive frames the mouth must stay open "
             "before flagging 'TALKING' (video/live only).",
    )

    conf_thresh = st.sidebar.slider(
        "Person Detection Confidence",
        min_value=0.25,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Minimum YOLO confidence to accept a person detection.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        """
        **Tips**
        - Good frontal lighting improves face-mesh accuracy.
        - Place camera so faces are ≥ 80 px tall.
        - For RTSP use: `rtsp://user:pass@ip:554/stream`
        """
    )

    return {
        "mode": mode,
        "mar_threshold": mar_thresh,
        "consecutive_frames": consec,
        "conf_threshold": conf_thresh,
    }


# ---------------------------------------------------------------------------
# Analytics helpers
# ---------------------------------------------------------------------------

def show_live_metrics(stats: Dict[str, Any]) -> None:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", stats.get("total_students", 0))
    c2.metric("Currently Talking", stats.get("talking", 0),
              delta=None, delta_color="inverse")
    c3.metric("Quiet / Attentive", stats.get("quiet", 0))
    c4.metric("Face Not Visible", stats.get("face_not_visible", 0))


def show_leaderboard(tracker: ClassroomTalkTracker, key_suffix: str = "") -> None:
    rows = tracker.get_leaderboard()
    if not rows:
        st.info("No talking data recorded yet.")
        return

    df = pd.DataFrame(rows)
    st.subheader("📊 Talking Leaderboard")
    st.dataframe(df, use_container_width=True, hide_index=True)

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Report (CSV)",
        data=csv,
        file_name=f"talking_report_{int(time.time())}.csv",
        mime="text/csv",
        key=f"csv_dl_{key_suffix}",
    )


# ---------------------------------------------------------------------------
# Mode: Image(s)
# ---------------------------------------------------------------------------

def run_image_mode(tracker: ClassroomTalkTracker, params: Dict) -> None:
    st.header("🖼️ Image Analysis")
    uploaded = st.file_uploader(
        "Upload classroom photo(s)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        accept_multiple_files=True,
    )

    if not uploaded:
        st.info("Upload one or more classroom images to begin.")
        return

    tracker.update_params(
        conf_threshold=params["conf_threshold"],
        mar_threshold=params["mar_threshold"],
        consecutive_frames=params["consecutive_frames"],
    )
    tracker.reset()

    for idx, file in enumerate(uploaded):
        st.markdown(f"#### {file.name}")
        img = load_image_safely(file)
        if img is None:
            st.error(f"Could not decode image: {file.name}")
            continue

        with st.spinner("Analyzing…"):
            annotated, stats = tracker.process_frame(img, is_image=True)

        col1, col2 = st.columns(2)
        with col1:
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Original", use_container_width=True)
        with col2:
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_container_width=True)

        show_live_metrics(stats)

        # Per-image student table
        if stats["students"]:
            rows = []
            for tid, s in stats["students"].items():
                rows.append({
                    "Student ID": f"Student #{tid}",
                    "Status": s["status"],
                    "MAR": s["mar"],
                    "Face Visible": s["face_visible"],
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.markdown("---")

    # Aggregate leaderboard across all images in this batch
    show_leaderboard(tracker, key_suffix="img")


# ---------------------------------------------------------------------------
# Mode: Video
# ---------------------------------------------------------------------------

def run_video_mode(tracker: ClassroomTalkTracker, params: Dict) -> None:
    st.header("🎬 Video Analysis")
    uploaded = st.file_uploader(
        "Upload classroom video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )

    if not uploaded:
        st.info("Upload a video file to begin processing.")
        return

    # Save to temp file (OpenCV needs a real path)
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
    try:
        tfile.write(uploaded.read())
        tfile.flush()
        video_path = tfile.name
    finally:
        tfile.close()

    tracker.update_params(
        conf_threshold=params["conf_threshold"],
        mar_threshold=params["mar_threshold"],
        consecutive_frames=params["consecutive_frames"],
    )
    tracker.reset()

    cap = None
    writer = None
    out_path = None

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Failed to open video. File may be corrupted or unsupported.")
            return

        info = get_video_info(cap)
        fps = info["fps"] if info["fps"] > 1 else 25.0
        tracker.fps = fps
        width, height = info["width"], info["height"]
        total_frames = info["frame_count"]

        st.write(
            f"**Resolution:** {width}×{height} &nbsp;|&nbsp; "
            f"**FPS:** {fps:.1f} &nbsp;|&nbsp; "
            f"**Frames:** {total_frames}"
        )

        # Output video
        out_path = tempfile.mktemp(suffix="_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

        progress = st.progress(0.0, text="Processing frames…")
        status_placeholder = st.empty()
        preview_placeholder = st.empty()
        metrics_placeholder = st.empty()

        frame_idx = 0
        last_stats: Dict[str, Any] = {}

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            annotated, stats = tracker.process_frame(frame, is_image=False)
            last_stats = stats

            if writer is not None:
                writer.write(annotated)

            frame_idx += 1
            if total_frames > 0:
                progress.progress(
                    min(frame_idx / total_frames, 1.0),
                    text=f"Frame {frame_idx}/{total_frames}",
                )

            # Update preview every ~15 frames to keep UI responsive
            if frame_idx % 15 == 0 or frame_idx == 1:
                preview_placeholder.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                    use_container_width=True,
                )
                with metrics_placeholder.container():
                    show_live_metrics(stats)

        progress.progress(1.0, text="Done!")
        status_placeholder.success(f"Processed {frame_idx} frames.")

        # Final metrics + leaderboard
        if last_stats:
            show_live_metrics(last_stats)
        show_leaderboard(tracker, key_suffix="vid")

        # Download button
        if out_path and os.path.exists(out_path):
            with open(out_path, "rb") as f:
                st.download_button(
                    label="⬇️ Download Annotated Video",
                    data=f.read(),
                    file_name=f"annotated_{uploaded.name}",
                    mime="video/mp4",
                    key="video_dl",
                )

    except Exception as e:
        st.error(f"Video processing error: {e}")
        logger.exception("Video mode failed")
    finally:
        safe_release(cap, writer)
        # Clean temp files
        for p in (video_path, out_path):
            if p and os.path.exists(p):
                try:
                    os.unlink(p)
                except OSError:
                    pass


# ---------------------------------------------------------------------------
# Mode: Live Camera / RTSP
# ---------------------------------------------------------------------------

def run_live_mode(tracker: ClassroomTalkTracker, params: Dict) -> None:
    st.header("📷 Live Monitoring")

    source_type = st.radio(
        "Source",
        ["Webcam (device index)", "RTSP / HTTP URL"],
        horizontal=True,
    )

    if source_type.startswith("Webcam"):
        cam_idx = st.number_input("Camera index", min_value=0, max_value=10, value=0, step=1)
        source = int(cam_idx)
    else:
        source = st.text_input(
            "Stream URL",
            value="rtsp://",
            help="Example: rtsp://admin:password@192.168.1.64:554/stream1",
        )
        if not source or source == "rtsp://":
            st.warning("Enter a valid RTSP or HTTP stream URL.")
            return

    start = st.button("▶️ Start Live Session", type="primary")
    stop_placeholder = st.empty()

    if not start:
        st.info("Configure source and press **Start Live Session**.")
        return

    tracker.update_params(
        conf_threshold=params["conf_threshold"],
        mar_threshold=params["mar_threshold"],
        consecutive_frames=params["consecutive_frames"],
    )
    tracker.reset()
    tracker.fps = 15.0  # conservative estimate for live

    cap = None
    try:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            st.error(
                "Could not open camera / stream. "
                "Check index, permissions, or RTSP credentials."
            )
            return

        # Reduce buffer for lower latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        preview = st.empty()
        metrics = st.empty()
        stop_btn = stop_placeholder.button("⏹️ Stop Session")

        frame_count = 0
        t0 = time.time()

        while not stop_btn:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.warning("Stream ended or frame grab failed. Retrying…")
                time.sleep(0.5)
                # Try to re-open once
                cap.release()
                cap = cv2.VideoCapture(source)
                if not cap.isOpened():
                    st.error("Stream permanently lost.")
                    break
                continue

            annotated, stats = tracker.process_frame(frame, is_image=False)
            frame_count += 1

            # Update FPS estimate
            elapsed = time.time() - t0
            if elapsed > 1.0:
                tracker.fps = frame_count / elapsed

            preview.image(
                cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                channels="RGB",
                use_container_width=True,
            )
            with metrics.container():
                show_live_metrics(stats)

            # Small sleep to keep Streamlit responsive
            time.sleep(0.01)
            stop_btn = stop_placeholder.button("⏹️ Stop Session", key=f"stop_{frame_count}")

        st.success("Live session ended.")
        show_leaderboard(tracker, key_suffix="live")

    except Exception as e:
        st.error(f"Live mode error: {e}")
        logger.exception("Live mode failed")
    finally:
        safe_release(cap)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    st.title("🎓 Classroom Talking Detector")
    st.markdown(
        "Detect and track students talking in real time using **YOLO** multi-object "
        "tracking + **MediaPipe Face Mesh** Mouth Aspect Ratio analysis."
    )

    params = render_sidebar()

    # Load (or retrieve cached) tracker
    try:
        tracker = load_tracker(
            conf=params["conf_threshold"],
            mar=params["mar_threshold"],
            consec=params["consecutive_frames"],
        )
        # Keep runtime params in sync even if cache returns old instance
        tracker.update_params(
            conf_threshold=params["conf_threshold"],
            mar_threshold=params["mar_threshold"],
            consecutive_frames=params["consecutive_frames"],
        )
    except Exception as e:
        st.error(
            f"Failed to initialize models: {e}\n\n"
            "Ensure you have internet on first run so YOLO weights can download."
        )
        st.stop()

    mode = params["mode"]
    if mode.startswith("🖼️"):
        run_image_mode(tracker, params)
    elif mode.startswith("🎬"):
        run_video_mode(tracker, params)
    else:
        run_live_mode(tracker, params)

    st.markdown("---")
    st.caption(
        "Built with Ultralytics YOLO • MediaPipe Face Mesh • Streamlit • OpenCV  |  "
        "Zero-crash design with graceful face-occlusion handling"
    )


if __name__ == "__main__":
    main()
