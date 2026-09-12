# Virtual Smoking

Real-time augmented reality application that creates an illusion of smoking using webcam, OpenCV, and MediaPipe.

The system tracks the user's face and hands, renders a virtual cigarette between their fingers, and detects smoking-like patterns (inhalation/exhalation) using deterministic rule-based logic. When an inhalation pattern is recognized, the cigarette ember glows. When an exhalation pattern follows, virtual smoke particles drift from the mouth.

> **Important:** This system does **not** use machine learning, AI, deep learning, neural networks, or model training. Smoking actions are inferred from predefined visual patterns and temporal landmark measurements using pure geometry, thresholds, and rule-based state machines.

---

## Installation

```bash
# 1. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download MediaPipe model files (required)
curl -sL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o face_landmarker.task
curl -sL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
```

**Requirements:**
- Python 3.11-3.13
- Webcam
- macOS/Linux/Windows

---

## Running

```bash
# Production mode (clean AR experience)
python src/main.py

# Debug mode (shows all tracking data and state info)
python src/main.py --debug
# or
python src/main.py -d

# Force the portable 2D renderer
python src/main.py --2d

# Render a 3D asset preview without opening the webcam
python scripts/render_3d_preview.py
```

**Camera permission:** On macOS, grant camera access in System Settings → Privacy & Security → Camera.

---

## Controls

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit application |
| `D` / `d` | Toggle debug mode (production ↔ debug) |
| `3` | Toggle between 3D and 2D when 3D is available |

---

## How It Works

### Pipeline

```text
Webcam
  ↓
Face Tracking (MediaPipe FaceLandmarker)
  ↓
Hand Tracking (MediaPipe HandLandmarker)
  ↓
Cigarette Tracking (thumb-index geometry)
  ↓
Mouth Interaction (distance + approach states)
  ↓
Smoking State Machine (8 states)
  ↓
Cigarette Glow (fade in/out on INHALING)
  ↓
Exhalation Detection (mouth pattern + cigarette away)
  ↓
Smoke Particles (spawn once per exhalation)
  ↓
Final AR Frame
```

### State Machine (8 States)

```
IDLE
  → APPROACHING (3 frames cigarette approaching mouth)
    → NEAR_MOUTH (3 frames cigarette near mouth)
      → INHALATION_CANDIDATE (4 frames mouth pattern change)
        → INHALING (4 frames confirmed pattern) ← GLOW ACTIVE
          → EXHALATION_CANDIDATE (3 frames cigarette moving away)
            → EXHALING (5 frames exhalation mouth pattern) ← SMOKE SPAWNS
              → COMPLETED (2 frames)
                → IDLE
```

### Detection Logic

- **No ML/AI**: All detection uses geometric calculations (distance, angle, velocity) and temporal frame counting
- **Inhalation**: Detected when cigarette is near mouth AND mouth opening/aspect ratio changes significantly for multiple consecutive frames
- **Exhalation**: Only becomes possible AFTER a confirmed inhalation, when cigarette moves away AND mouth shows exhalation-like pattern
- **Hysteresis**: Separate entry/exit thresholds prevent flickering (e.g., NEAR enters at 80px, exits at 96px)

### Visual Effects

1. **Virtual Cigarette**: GLB model with a PNG-based 2D fallback, follows the thumb-index midpoint and finger orientation
2. **Ember Glow**: Emissive 3D material or PNG fallback, fades in/out smoothly when INHALING state is active
3. **Smoke Particles**: 8-16 particles per exhalation, expand, drift upward, fade out over 30-60 frames

---

## Configuration

All tunable parameters are centralized in `src/config.py`:

```python
Config.CAMERA              # Camera settings
Config.FACE_TRACKER        # Face tracking + mouth smoothing
Config.HAND_TRACKER        # Hand tracking
Config.CIGARETTE_TRACKER   # Cigarette position/rotation smoothing (OneEuroFilter)
Config.CIGARETTE_MOUTH_DETECTOR  # Distance thresholds, hysteresis, frame counts
Config.SMOKING_DETECTOR    # Inhalation/exhalation thresholds, frame windows
Config.GLOW_EFFECT         # Glow fade speeds
Config.SMOKE_EFFECT        # Particle counts, physics, origin offset
Config.DEBUG               # Debug display settings
```

**Key thresholds to tune:**
- `CIGARETTE_MOUTH_DETECTOR['near_threshold']` (default: 80px)
- `SMOKING_DETECTOR['mouth_opening_change_threshold']` (default: 4px)
- `SMOKING_DETECTOR['exhalation_mouth_opening_threshold']` (default: 6px)

---

## Project Structure

```
virtual-smoking/
│
├── src/
│   ├── main.py                      # Entry point, app orchestration
│   │
│   ├── camera/
│   │   └── camera.py                # Webcam capture, mirror, FPS
│   │
│   ├── vision/
│   │   ├── face_tracker.py          # MediaPipe face + mouth measurements
│   │   └── hand_tracker.py          # MediaPipe hand landmarks
│   │
│   ├── interaction/
│   │   ├── cigarette_tracker.py     # Cigarette pos/rot from fingers
│   │   ├── cigarette_mouth_detector.py  # Distance + approach states
│   │   └── smoking_detector.py      # 8-state FSM (inhale + exhale)
│   │
│   ├── effects/
│   │   ├── cigarette.py             # Cigarette PNG renderer + fallback
│   │   ├── cigarette_3d.py          # ModernGL GLB renderer
│   │   ├── glow.py                  # Ember glow with fade
│   │   └── smoke.py                 # Particle system
│   │
│   └── utils/
│       ├── geometry.py              # Distance, angle, midpoint, rotation
│       └── smoothing.py             # Smoother, OneEuroFilter, AngleOneEuroFilter
│
├── assets/
│   ├── cigarette/
│   │   ├── cigarette.glb            # 3D cigarette model
│   │   ├── cigarette.png            # 180x30 RGBA
│   │   └── cigarette_glow.png       # Radial ember gradient
│   │
│   └── smoke/                       # (placeholder)
│
├── tests/
│   └── test_utils.py                # Unit tests for geometry/smoothing
│
├── requirements.txt
├── README.md
├── .gitignore
└── src/config.py                    # All tunable parameters
```

---

## Asset Validation

At startup, the application verifies these required files exist:
- `face_landmarker.task`
- `hand_landmarker.task`
- `assets/cigarette/cigarette.png`
- `assets/cigarette/cigarette_glow.png`

The GLB is optional: when it or the 3D dependencies are unavailable, the application starts with the 2D renderer.

---

## Performance

- **Target:** 30+ FPS on typical hardware
- **Optimization:** OneEuroFilter for responsive yet stable tracking; particle lifecycle cleanup prevents memory growth; all resources initialized once

---

## Testing

```bash
# Run unit tests
python tests/test_utils.py
python tests/test_unit.py
python tests/test_startup.py
python tests/test_cigarette_3d.py
```

Tests cover:
- Geometry functions (distance, midpoint, angle, clamp)
- Smoothing (Smoother, OneEuroFilter, AngleOneEuroFilter, OneEuroFilter2D)
- Cigarette tracker
- Cigarette-mouth detector
- Smoking detector (state machine)
- Smoke particle system
- Startup paths and command-line options
- GLB material bindings and pixel-aligned 3D transforms

---

## Evaluation

### Test Suite
The project includes a comprehensive evaluation framework in `tests/`:

```bash
# Run unit tests
python tests/test_unit.py

# Run FPS benchmark (30 seconds)
python tests/benchmark_fps.py --duration 30

# Run stress test (5 minutes)
python tests/stress_test.py --duration 5

# Generate evaluation report
python tests/run_evaluation.py
```

### Manual Testing Protocol
See `tests/MANUAL_TEST_PROTOCOL.md` for standardized test procedures covering:
- Normal smoking sequences (positive tests)
- False positive scenarios (talking, smiling, random movement)
- Tracking robustness (head/hand movement, distance, recovery)
- Environmental conditions (lighting, backgrounds)
- 20-cycle repeatability test
- 5-minute stress test

### Validation Checklist
See `tests/VALIDATION_CHECKLIST.md` for complete validation criteria.

### Metrics & Analysis
```bash
# Evaluate confusion matrix from test results
python tests/evaluate_confusion.py
```

Results are saved to `results/` directory:
- `evaluation_results.json` - Raw evaluation data
- `EVALUATION_REPORT.md` - Formatted report
- `benchmark_fps.json` - FPS benchmark results
- `stress_test_results.json` - Stress test metrics

---

## Limitations

- **Webcam quality**: Low light or low resolution reduces landmark accuracy
- **Lighting**: Strong backlighting or shadows can cause tracking loss
- **Hand occlusion**: Fingers must be visible for cigarette tracking
- **Mouth detection**: Requires clear face view; masks/beards may interfere
- **Visual only**: Detection is based on predefined visual patterns, not physiological breathing
- **Single user**: Tracks one face and one hand at a time
- **Approximate depth**: The 3D cigarette uses estimated screen-space placement and does not occlude behind fingers

---

## License

MIT License - Feel free to use and modify.
