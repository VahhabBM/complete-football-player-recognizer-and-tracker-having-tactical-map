# complete-football-player-recognizer-and-tracker-having-tactical-map
An AI-powered football analytics tool that tracks players, referees, and the ball using YOLO, computes a real-time dynamic homography matrix based on pitch keypoints, and projects tactical movements onto a 2D mini-map radar.

# Football Tactical Radar & Player Tracker ⚽🏃‍♂️📊

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-vibrant.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange.svg)](https://opencv.org/)

An advanced, AI-powered football analytics and computer vision pipeline. This project detects and tracks players, referees, and the ball from broadcast or tactical match footage, automatically maps the field's keypoints to compute a dynamic Homography matrix, and projects everyone's absolute positions onto a moving 2D tactical mini-map (radar) in real-time.

---

## 🌟 Key Features

* **Dual-Model Deep Learning Pipeline:** Utilizes a specialized object detection model for tracking entities (players, referees, ball) and a regression/pose model for pinpointing pitch keypoints.
* **Dynamic Robust Homography:** Computes continuous perspective transformations using OpenCV's RANSAC, allowing the system to adapt smoothly to camera panning, tilting, and zooming.
* **Dual-Rotation Correction (Two-Pass Alignment):** Includes a manual/deterministic geometric override that corrects both horizontal orientation (180° inversion) and vertical mirroring simultaneously, ensuring perfect tactical alignment regardless of camera angles.
* **Static Stability Fallback:** Solves the classic collinearity / blind-spot problem (e.g., when fewer than 4 keypoints are detected or when points align on a single straight line). The pipeline drops corrupted matrices and locks onto the last valid wide-angle frame buffer, preventing the tactical dots from squeezing onto a single line.
* **Automatic Team Clustering:** Dynamic color-space accumulation that samples player jerseys during a warm-up phase to automatically assign team IDs and handle referee isolation.
* **Alpha-Blended Tactical Overlay:** Real-time generation of a semi-transparent 2D tactical mini-map blended onto the output video frame.

---

## 🏗️ System Architecture & Pipeline

```text
[ Input Match Video ]
│
├──► [ Object Detection & Tracking ] ──► Player, Goalkeeper, Referee, Ball Bounding Boxes
│                                              │
│                                              ▼
├──► [ Pitch Pose Estimation ]       ──► 9 Strategic Keypoints (Center Spine & Right Wing)
│         │                                    │
│         ▼                                    ▼
│   [ Geometric Validation ] ─────────► [ Dynamic Homography Computation ]
│         │                                    │ (Deterministic 2-Pass Refinement)
│         ▼                                    ▼
│   [ Stability Fallback ]   ─────────► [ Coordinates Transformation ]
│                                              │ (Project Foot Coordinates to 2D Plane)
│                                              ▼
└─────────────────────────────────────► [ Alpha-Blended Visual Renderer ] ──► [ Output Video ]
```

---

## 📁 Project Structure

```text
├── data/
│   ├── input/               # Place your input match.mp4 here
│   └── output/              # Final processed video with the tactical radar
├── runs/
│   ├── detect/              # YOLO weights for player and object tracking
│   └── pose/                # YOLO weights for pitch keypoints detection
├── tracking/                # Object tracking wrappers
├── team_assignment/         # Color clustering and team isolation logic
├── transformation/
│   └── transformer.py       # Core Homography, Dual-Rotation, and Stability Fallback
├── visualization/
│   └── renderer.py          # Field blueprint drafting and tactical alpha-blending
└── main.py                  # Project entry-point and execution pipeline
```

⚡ Getting Started
1. Prerequisites
Make sure you have Python 3.10+ installed. It is highly recommended to use a virtual environment (venv or conda).

2. Installation
Clone the repository and install the required dependencies:

```text
git clone [https://github.com/YOUR_USERNAME/football-tactical-radar.git](https://github.com/YOUR_USERNAME/football-tactical-radar.git)
cd football-tactical-radar
pip install -r requirements.txt
```

(Note: Ensure your requirements.txt contains ultralytics, opencv-python, and numpy)

3. Model Weights & Data Setup:

    1. Place your fine-tuned player detection weights into runs/detect/.

    2.Place your field keypoint pose weights into runs/pose/.

    3.Put your source video file into data/input/match.mp4.

4. Running the Pipeline
Execute the main orchestration script:
```text
python main.py
```
The system will display a progress bar. Once completed, the final annotated video with the embedded tactical radar will be saved under data/output/final_output.mp4.

🛠️ Deep Dive: The Transformation Engine
The structural backbone of this project lies within transformation/transformer.py. It tackles two major challenges in sports telemetry:

1. The Blind Spot & Edge Case Solution (Collinearity)
When the broadcast camera pans rapidly or zooms into a tight duel, the number of visible keypoints drops below 4, or they align on a single straight line (e.g., the midfield line). Mathematically, computing a projection matrix from collinear points causes the determinant to collapse, flattening all players onto a single line on the radar.
Our Static Stability Fallback detects this drop in polygon area/determinant and instantly rejects the corrupt matrix, utilizing a steady fallback buffer from the last healthy perspective.

2. Double-Pass Geometric Correction
Depending on which side of the stadium the camera is facing, the mathematical matrix can easily flip the coordinate directions. The pipeline enforces a two-pass correction:

    1. Horizontal Pass: Rotates the field by 180° to align tracking direction with gameplay length.
    2. 2.Vertical Pass: Flips the width axis using an affine reflection matrix around the central pitch horizon ($Y = 340$).

🎯 Sample Results
Broadcast View with Trackers & Keypoints  Tactical 2D Mini-Map Radar Output
Object Bounding Boxes + Active Hotspots  Real-Time Projected Dots & Ball Vectors

## 👥 Contributors

This project was developed collaboratively. Special thanks to my co-developer for their contributions to the pipeline architecture and geometry logic:

* **Arash** - [@araswithh](https://github.com/araswithh)
