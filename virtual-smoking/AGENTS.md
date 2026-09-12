# Virtual Smoking - AGENTS.md

## Project Overview
Real-time AR cigarette/smoke effect using webcam + MediaPipe landmarks. No ML/AI - pure geometry, thresholds, and rule-based state machines.

**Tech Stack:** Python 3.11+, OpenCV, MediaPipe 0.10.30 (Tasks API), NumPy, ModernGL (3D rendering)

---

## Setup & Run

```bash
cd virtual-smoking
source venv/bin/activate
python src/main.py
```

**Camera permission required on macOS:** System Settings → Privacy & Security → Camera

---

## Controls (in running app)

| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `D` / `d` | Toggle debug mode (production ↔ debug) |
| `3` | Toggle 3D/2D rendering |

---

## Project Structure (Phases 1-14 Complete)

```
src/
├── main.py                          # Entry point, pipeline orchestration
├── camera/
│   └── camera.py                    # Webcam capture, mirror, FPS
├── vision/
│   ├── face_tracker.py              # MediaPipe FaceLandmarker (Tasks API)
│   └── hand_tracker.py              # MediaPipe HandLandmarker (Tasks API)
├── interaction/
│   ├── cigarette_tracker.py         # Cigarette pos/rot from thumb-index
│   ├── cigarette_mouth_detector.py  # Distance + approach/near/away states
│   └── smoking_detector.py          # 8-state FSM (inhale + exhale)
├── effects/
│   ├── cigarette.py                 # 2D Cigarette PNG renderer + fallback
│   ├── cigarette_3d.py              # 3D GLB renderer (ModernGL + PBR)
│   ├── glow.py                      # Ember glow (fade in/out) - 2D fallback
│   └── smoke.py                     # Particle system (spawn on exhale)
└── utils/
    ├── geometry.py                  # Distance, angle, midpoint, rotation
    └── smoothing.py                 # Smoother, OneEuroFilter, OneEuroFilter2D, AngleOneEuroFilter

assets/
├── cigarette/
│   ├── cigarette.glb                # 3D model (PBR, 118 verts, 4 textures)
│   ├── cigarette.png                # 180x30 RGBA (2D fallback)
│   └── cigarette_glow.png           # Radial ember gradient (2D fallback)
└── smoke/

tests/
├── test_utils.py                    # Geometry + smoothing tests
├── test_unit.py                     # All component unit tests
├── benchmark_fps.py                 # FPS benchmarking
├── stress_test.py                   # 5-min stability test
├── run_evaluation.py                # Evaluation report generator
├── evaluate_confusion.py            # Confusion matrix analysis
├── MANUAL_TEST_PROTOCOL.md          # Standardized testing procedures
├── VALIDATION_CHECKLIST.md          # Complete validation criteria
└── evaluation/
    └── metrics.py                   # Event logging, metrics, confusion matrix

results/                             # Generated evaluation outputs
```

---

## MediaPipe Models (Required)

Download to project root (gitignored):
```bash
curl -sL https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task -o face_landmarker.task
curl -sL https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task -o hand_landmarker.task
```

---

## State Machine (Phases 7-9)

```
IDLE
  → APPROACHING (3 frames cig=APPROACHING)
    → NEAR_MOUTH (3 frames cig=NEAR)
      → INHALATION_CANDIDATE (4 frames mouth change while NEAR)
        → INHALING (4 frames pattern confirmed while NEAR)
          → EXHALATION_CANDIDATE (3 frames cig=MOVING_AWAY after INHALING)
            → EXHALING (5 frames mouth pattern while AWAY)
              → COMPLETED (2 frames)
                → IDLE
```

**Glow:** Active only in `INHALING` state (Phase 8/14 - built into 3D renderer)

**Smoke:** Spawns once on rising edge of `EXHALING` state (Phase 10)

---

## Key Configuration (tunable in code)

| Class | Parameters |
|-------|------------|
| `CigaretteMouthDetector` | `near_threshold=80`, `approach_frames=3`, `near_frames=3` |
| `SmokingDetector` | `INHALATION_FRAME_COUNT=4`, `EXHALATION_FRAME_COUNT=5`, `EXHALATION_MOUTH_OPENING_THRESHOLD=6` |
| `GlowEffect` | `fade_in_speed=0.15`, `fade_out_speed=0.08` |
| `SmokeEffect.config` | `particle_count_min/max=8/16`, `lifetime=30-60`, `upward_force=0.08` |

**3D Renderer Config (`cigarette_3d.py`):**
| Parameter | Default | Notes |
|-----------|---------|-------|
| `model_scale` | `1.0` | Was 0.005 (microscopic) |
| `model_offset_y` | `-0.1` | Vertical offset |
| `model_rotation_offset_y` | `np.pi` | 180° to face camera |
| `model_rotation_offset_x` | `0.0` | No X tilt |
| `glow_fade_in/out` | `0.15`/`0.08` | Matches GlowEffect |

---

## Common Gotchas

1. **MediaPipe 1.x uses Tasks API** - not `mp.solutions`. Use `mediapipe==0.10.30`
2. **Model files** must be in working directory (not `src/`)
3. **Camera index 0** may fail on macOS without permission grant
4. **Standalone test scripts** - no separate test runner is required
5. **Git ignores:** `venv/`, `*.task`, `create_*.py`, `__pycache__/`, `assets/cigarette/textures/`
6. **3D rendering issues fixed:**
   - Model scale was 0.005 (microscopic) → **1.0**
   - Depth -2.0 → **-1.5** (closer)
   - Rotation X=-π/2 → **X=0, Y=π** (faces camera)
   - World position multiplier 1.5 → **2.0**
   - GLSL `bool` uniform → **`int`** (GLSL has no native bool)
   - Normal map uniform removed from shader (was breaking lighting)
   - View matrix fixed in `set_view_projection` with proper `look_at`

---

## Running Tests

```bash
# Unit tests (no webcam)
python tests/test_unit.py
python tests/test_utils.py
python tests/test_startup.py
python tests/test_cigarette_3d.py

# FPS benchmark (30s)
python tests/benchmark_fps.py --duration 30

# Stress test (5 min)
python tests/stress_test.py --duration 5

# Generate evaluation report
python tests/run_evaluation.py
```

---

## Running the App

```bash
# Production mode (clean AR)
python src/main.py

# Debug mode (all overlays)
python src/main.py --debug

# Force 2D rendering
python src/main.py --2d

# Render the GLB to output/cigarette_3d_preview.png
python scripts/render_3d_preview.py

# Controls:
#   q/ESC - Quit
#   D/d - Toggle debug mode
#   3 - Toggle 3D/2D rendering
```

---

## Architecture Notes

**Pipeline:**
```
Webcam → OpenCV → MediaPipe → Hand/Face Landmarks → Cigarette Geometry → 3D/2D Transform → Rendering → Smoking State Machine → Glow (3D internal / 2D separate) → Smoke Particles → Final AR Frame
```

**3D Renderer (`cigarette_3d.py`):**
- Loads GLB with PBR textures (albedo, metallic/roughness, emissive)
- ModernGL offscreen framebuffer → alpha composite onto webcam frame
- PBR vertex/fragment shaders with emissive glow
- Model transform from 2D hand tracking (pos, rot) → 3D world transform
- Configurable model offset, rotation, scale
- Built-in glow via emissive material (fade in/out)

**2D Fallback (`cigarette.py`, `glow.py`):**
- PNG overlay with alpha blending
- Separate glow renderer for 2D mode

**No ML/AI** - All detection uses geometric calculations + temporal frame counting.

---

## Key Configuration Files

- `src/config.py` - All tunable parameters centralized
- `requirements.txt` - Python dependencies
- `.gitignore` - Excludes venv, models, textures, create_*.py

---

## Adding New Phases

Follow existing pattern:
1. Create module in appropriate `src/` subdirectory
2. Import in `main.py`, instantiate, call `update()` in loop
3. Add debug toggle key if needed
4. Keep state transitions deterministic, frame-count based
5. Centralize thresholds as class attributes (no magic numbers)
