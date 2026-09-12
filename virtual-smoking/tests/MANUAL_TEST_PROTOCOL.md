# Virtual Smoking - Manual Testing Protocol

## Overview
This document defines the standardized manual testing protocol for evaluating the Virtual Smoking AR application.

## Test Environment
- **Application**: Virtual Smoking (Phases 1-12)
- **Mode**: Debug mode enabled (`python src/main.py --debug`)
- **Camera**: Default webcam (index 0)
- **Resolution**: 1280x720 @ 30 FPS target
- **Lighting**: Standard indoor lighting (test multiple conditions)

---

## Test 1: Normal Smoking Sequence (Positive Test)

### Procedure
1. Start application in debug mode
2. Position face and hand in view
3. Form pinch gesture (thumb + index) to "hold" cigarette
4. Move cigarette toward mouth
5. Hold near mouth, slightly open mouth (inhale pattern)
6. Keep cigarette near mouth for ~2 seconds
7. Move cigarette away from mouth
8. Open mouth wider (exhale pattern)
9. Observe smoke particles
10. Return to idle
11. Repeat 5x

### Expected Results
| Step | State | Glow | Smoke |
|------|-------|------|-------|
| Approach | APPROACHING | Off | Off |
| Near mouth | NEAR_MOUTH | Off | Off |
| Inhale pattern | INHALATION_CANDIDATE → INHALING | **Fade In** | Off |
| Move away | EXHALATION_CANDIDATE | Fade Out | Off |
| Exhale pattern | EXHALING | Off | **Spawn** |
| Complete | COMPLETED → IDLE | Off | Fade |

### Pass Criteria
- [ ] Cigarette appears between fingers
- [ ] Cigarette tracks hand smoothly
- [ ] Inhalation detected (state shows INHALING)
- [ ] Glow fades in smoothly
- [ ] Exhalation detected (state shows EXHALING)
- [ ] Smoke particles spawn once
- [ ] Smoke fades naturally
- [ ] Returns to IDLE
- [ ] Cycle repeatable

---

## Test 2: False Positive Scenarios (Negative Tests)

### 2A: Talking
- Speak normally for 30 seconds
- Move mouth naturally
- **Expected**: No INHALING, No EXHALING, No smoke

### 2B: Smiling/Laughing
- Smile broadly, laugh
- **Expected**: No false triggers

### 2C: Normal Mouth Opening
- Open/close mouth without cigarette near
- **Expected**: No triggers

### 2D: Hand Near Face (No Cigarette)
- Move hand near mouth without pinch gesture
- **Expected**: No cigarette, no triggers

### 2E: Cigarette Near Mouth, No Inhale
- Bring cigarette to mouth, hold still
- Don't perform inhale pattern
- **Expected**: State NEAR_MOUTH, no INHALING

### 2F: Cigarette Away, No Exhale
- Perform inhalation, then move cigarette away
- Don't perform exhale pattern
- **Expected**: State EXHALATION_CANDIDATE, no EXHALING, no smoke

### 2G: Fast Random Movement
- Wave hand quickly near face
- Move head rapidly
- **Expected**: No false triggers, stable tracking

---

## Test 3: Tracking Robustness

### 3A: Head Movement
- Move head: Left, Right, Up, Down, Tilt
- **Expected**: Cigarette stable, mouth tracking stable

### 3B: Hand Movement Speed
- Slow, Medium, Fast hand movements
- **Expected**: Cigarette follows without teleporting

### 3C: Distance from Camera
- Close (30cm), Normal (60cm), Far (100cm)
- **Expected**: Tracking works at all distances

### 3D: Tracking Loss & Recovery
- Move face out of frame → return
- Move hand out of frame → return
- Both out → return
- **Expected**: No crash, clean recovery, state resets

---

## Test 4: Environmental Conditions

| Condition | Notes |
|-----------|-------|
| Good lighting (daylight) | Baseline |
| Low lighting (dim room) | Test tracking quality |
| Bright lighting (direct sun) | Test overexposure |
| Busy background | Test segmentation |
| Plain background | Test tracking stability |

---

## Test 5: Repeated Cycle Test (20 Cycles)

Perform complete smoking sequence 20 times. Record:

| Cycle | Inhale | Exhale | Smoke | False | Notes |
|-------|--------|--------|-------|-------|-------|
| 1     |        |        |       |       |       |
| 2     |        |        |       |       |       |
| ...   |        |        |       |       |       |
| 20    |        |        |       |       |       |

**Metrics**:
- Inhalation detection rate: X/20
- Exhalation detection rate: X/20
- Smoke trigger rate: X/20
- False positive rate: X/20

---

## Test 5: Stress Test (5 Minutes Continuous)

Run application continuously for 5 minutes performing repeated cycles.

**Check**:
- [ ] FPS stable (no degradation >10%)
- [ ] Memory stable (no growth)
- [ ] No particle accumulation
- [ ] No state machine freezes
- [ ] No crashes
- [ ] Tracking recovers from losses

---

## Test 6: Latency Estimation

Using debug mode timestamps:
1. Note frame when visual pattern occurs
2. Note frame when state changes
3. Note frame when effect appears

**Measure**:
- Inhalation pattern → INHALING state: frames
- INHALING state → Glow visible: frames
- Exhale pattern → EXHALING state: frames
- EXHALING state → Smoke spawn: frames

---

## Recording Results

For each test, record:
- Date/Time
- Lighting condition
- Camera distance
- Pass/Fail
- Observations
- Any errors/crashes

---

## Scoring Summary

| Test Category | Weight | Score |
|---------------|--------|-------|
| Normal Sequence | 30% | X% |
| False Positive | 25% | X% |
| Tracking Robustness | 20% | X% |
| Environmental | 10% | X% |
| Repeated Cycles | 10% | X% |
| Stress Test | 5% | X% |
| **Total** | **100%** | **X%** |

**Pass Threshold**: 80% overall, 100% on critical (no crashes, no permanent state corruption)