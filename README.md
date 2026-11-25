# Classroom Talking Detector

Production-ready computer-vision web application that detects and tracks students talking in a classroom across **images**, **recorded videos**, and **live camera / RTSP feeds**.

Built with:

- **Ultralytics YOLO** (YOLOv8 / YOLO11) + ByteTrack for persistent multi-student tracking
- **MediaPipe Face Mesh** for precise Mouth Aspect Ratio (MAR) lip-movement analysis
- **Streamlit** for a clean, interactive web UI
- **OpenCV** for video I/O and visualization

## Features

| Feature | Description |
|---------|-------------|
| Multi-student tracking | Persistent `Student #ID` across frames using YOLO `model.track()` |
| Talking detection | MAR (Mouth Aspect Ratio) with configurable threshold |
| Temporal smoothing | Requires N consecutive open-mouth frames before flagging “TALKING” (video/live) |
| Visual overlays | Green = Quiet, Red = Talking, Orange = Face Not Visible |
| Analytics dashboard | Live counters + talking leaderboard with CSV export |
| Zero-crash design | Graceful handling of occluded faces, corrupted frames, disconnected streams |
| Input modes | Single/batch images, video upload + download, webcam & RTSP |

## Project Structure

```
classroom_talking_detector/
├── app.py              # Streamlit web application
├── tracker.py          # YOLO tracking + MediaPipe MAR engine
├── utils.py            # MAR math, drawing, validation helpers
├── requirements.txt    # Pinned dependencies
└── README.md
```

## Installation

### 1. Clone / copy the project

```bash
cd classroom_talking_detector
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** YOLO weights (`yolo11n.pt` or `yolov8n.pt`) are downloaded automatically by Ultralytics when the model is first loaded. An internet connection is required only on the very first launch.

### 4. Launch the app

```bash
streamlit run app.py
```

The browser will open at `http://localhost:8501`.

## Usage Guide

### Sidebar Controls

- **MAR Sensitivity** (0.15 – 0.75, default 0.35)  
  Higher values = only wide-open mouths trigger detection. Lower values catch subtle talking.

- **Consecutive Frames** (1 – 30, default 6)  
  Number of successive frames the mouth must remain open before a student is marked “TALKING”. Only applies to video and live modes.

- **Person Detection Confidence** (0.25 – 0.90, default 0.40)  
  YOLO confidence threshold. Raise it in crowded scenes to reduce false person detections.

### Image Mode

1. Select **Image(s)** in the sidebar.
2. Upload one or more classroom photos (`.jpg`, `.png`, `.webp`, …).
3. Each image is annotated with bounding boxes, Student IDs, MAR scores and status labels.
4. A summary table and downloadable CSV report are shown at the bottom.

### Video Mode

1. Select **Video**.
2. Upload a classroom recording (`.mp4`, `.avi`, `.mov`, …).
3. The app processes every frame, shows a live preview, and produces an annotated video.
4. Download the annotated video and the CSV talking report when finished.

### Live Camera / RTSP

1. Select **Live Camera / RTSP**.
2. Choose Webcam (device index) or paste an RTSP/HTTP URL.
3. Click **Start Live Session**.
4. Real-time overlays and metrics update continuously.
5. Click **Stop Session** to end and view the final leaderboard.

## Optimal Camera & Lighting Tips

- **Face size**: Students’ faces should be at least ~80 px tall in the frame for reliable Face Mesh landmarks.
- **Angle**: Prefer frontal or near-frontal views. Extreme profile views are marked “Face Not Visible” and still keep body tracking.
- **Lighting**: Even, front lighting works best. Strong backlight or very dark rooms reduce landmark confidence.
- **Camera placement**: Mount slightly above eye level, centered on the classroom, covering all seats without extreme wide-angle distortion.
- **Frame rate**: 15–30 FPS is ideal. The temporal smoother adapts to the measured FPS.

## How Talking Detection Works

1. **YOLO + ByteTrack** detects every person and assigns a persistent numeric ID.
2. **MediaPipe Face Mesh** extracts 468 facial landmarks on the full frame.
3. Landmarks are associated to each student track via IoU / center-distance matching.
4. **Mouth Aspect Ratio** is computed:

   ```
   MAR = (vertical distance between inner upper & lower lip)
         / (horizontal distance between mouth corners)
   ```

5. In video/live mode a student is flagged **TALKING** only after the MAR stays above the threshold for N consecutive frames. This eliminates false positives from yawns, smiles, or single-frame noise.
6. In image mode a single-frame decision produces the label **LIKELY TALKING**.

If a face is turned away or occluded, the system never crashes — the student is simply labelled **Face Not Visible** while body tracking continues.

## Requirements (pinned)

See `requirements.txt`. Key packages:

- `streamlit==1.40.1`
- `opencv-python-headless==4.10.0.84`
- `ultralytics==8.3.49`
- `mediapipe==0.10.18`
- `numpy==1.26.4`
- `pandas==2.2.3`
- `Pillow==11.0.0`
- `torch` / `torchvision` (CPU or CUDA)

## Troubleshooting

| Issue | Solution |
|-------|----------|
| YOLO weights download fails | Manually place `yolo11n.pt` or `yolov8n.pt` in the working directory |
| Webcam not opening | Try a different camera index (0, 1, 2…) or check OS permissions |
| RTSP connection refused | Verify credentials, IP, port and that the stream is H.264 |
| Low FPS on CPU | Use a smaller YOLO model or reduce input resolution |
| Many “Face Not Visible” | Improve lighting / camera angle; faces must be reasonably frontal |

## License

This project is provided as-is for educational and research use. Respect local privacy regulations when deploying camera-based monitoring systems.
