# Virtual Smoking

Real-time augmented reality application that creates an illusion of smoking using webcam, OpenCV, and MediaPipe.

## Installation

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Running

```bash
python src/main.py
```

## Project Structure

```
virtual-smoking/
├── src/
│   ├── main.py                 # Entry point
│   ├── camera/camera.py        # Webcam handling
│   ├── vision/
│   │   ├── face_tracker.py     # MediaPipe face landmarks
│   │   └── hand_tracker.py     # MediaPipe hand landmarks
│   ├── interaction/
│   │   ├── cigarette_tracker.py    # Cigarette position/orientation
│   │   └── smoking_detector.py     # Pattern recognition state machine
│   ├── effects/
│   │   ├── cigarette.py        # Cigarette rendering
│   │   ├── glow.py             # Ember glow effect
│   │   └── smoke.py            # Smoke particle effect
│   └── utils/
│       ├── geometry.py         # Geometric calculations
│       └── smoothing.py        # Temporal smoothing
├── assets/
│   ├── cigarette/
│   │   ├── cigarette.png
│   │   └── cigarette_glow.png
│   └── smoke/
├── tests/
├── requirements.txt
└── README.md
```

## Current Phase

Phase 4: Combined face and hand tracking with landmark visualization.

## Controls

- `q` or `ESC` - Quit
- `d` - Toggle debug landmarks