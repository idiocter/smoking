# Virtual Smoking - Phase 13 Validation Checklist

**Date**: ___________
**Tester**: ___________
**Environment**: ___________

---

## Pre-Test Setup
- [ ] Camera connected and accessible
- [ ] MediaPipe models downloaded (`face_landmarker.task`, `hand_landmarker.task`)
- [ ] Assets present (`cigarette.png`, `cigarette_glow.png`)
- [ ] Virtual environment activated
- [ ] Dependencies installed

---

## Unit Tests
- [ ] `python tests/test_utils.py` - PASSED
- [ ] `python tests/test_unit.py` - PASSED

---

## Functional Tests

### Normal Smoking Sequence (5 runs)
| Run | Inhale Detected | Glow On | Exhale Detected | Smoke Spawned | Returns to IDLE | Notes |
|-----|----------------|---------|-----------------|---------------|-----------------|-------|
| 1   | ☐ Yes / ☐ No   | ☐       | ☐ Yes / ☐ No    | ☐             | ☐               |       |
| 2   | ☐ Yes / ☐ No   | ☐       | ☐ Yes / ☐ No    | ☐             | ☐               |       |
| 3   | ☐ Yes / ☐ No   | ☐       | ☐ Yes / ☐ No    | ☐             | ☐               |       |
| 4   | ☐ Yes / ☐ No   | ☐       | ☐ Yes / ☐ No    | ☐             | ☐               |       |
| 5   | ☐ Yes / ☐ No   | ☐       | ☐ Yes / ☐ No    | ☐             | ☐               |       |

**Pass Rate**: _____/5

---

### False Positive Tests
| Scenario | Tested | False Trigger | Notes |
|----------|--------|---------------|-------|
| Talking (30s) | ☐ | ☐ Yes / ☐ No | |
| Smiling/Laughing | ☐ | ☐ Yes / ☐ No | |
| Normal mouth opening | ☐ | ☐ Yes / ☐ No | |
| Hand near face (no pinch) | ☐ | ☐ Yes / ☐ No | |
| Cigarette near mouth, no inhale | ☐ | ☐ Yes / ☐ No | |
| Cigarette away, no exhale | ☐ | ☐ Yes / ☐ No | |
| Fast random movement | ☐ | ☐ Yes / ☐ No | |

**False Positive Rate**: _____/7

---

### Tracking Robustness
| Test | Tested | Stable | Notes |
|------|--------|--------|-------|
| Head: Left | ☐ | ☐ Yes / ☐ No | |
| Head: Right | ☐ | ☐ Yes / ☐ No | |
| Head: Up/Down | ☐ | ☐ Yes / ☐ No | |
| Head: Tilt | ☐ | ☐ Yes / ☐ No | |
| Hand: Slow | ☐ | ☐ Yes / ☐ No | |
| Hand: Fast | ☐ | ☐ Yes / ☐ No | |
| Distance: Close | ☐ | ☐ Yes / ☐ No | |
| Distance: Normal | ☐ | ☐ Yes / ☐ No | |
| Distance: Far | ☐ | ☐ Yes / ☐ No | |
| Face loss/recovery | ☐ | ☐ Yes / ☐ No | |
| Hand loss/recovery | ☐ | ☐ Yes / ☐ No | |

---

### Environmental Conditions
| Condition | Tested | Quality | Notes |
|-----------|--------|---------|-------|
| Good lighting | ☐ | ☐ Good / ☐ Fair / ☐ Poor | |
| Low lighting | ☐ | ☐ Good / ☐ Fair / ☐ Poor | |
| Bright lighting | ☐ | ☐ Good / ☐ Fair / ☐ Poor | |
| Busy background | ☐ | ☐ Good / ☐ Fair / ☐ Poor | |

---

### Repeated Cycles (20 cycles)
| Cycle | Inhale | Exhale | Smoke | False Trigger | Notes |
|-------|--------|--------|-------|---------------|-------|
| 1     | ☐      | ☐      | ☐     | ☐             |       |
| 2     | ☐      | ☐      | ☐     | ☐             |       |
| 3     | ☐      | ☐      | ☐     | ☐             |       |
| 4     | ☐      | ☐      | ☐     | ☐             |       |
| 5     | ☐      | ☐      | ☐     | ☐             |       |
| 6     | ☐      | ☐      | ☐     | ☐             |       |
| 7     | ☐      | ☐      | ☐     | ☐             |       |
| 8     | ☐      | ☐      | ☐     | ☐             |       |
| 9     | ☐      | ☐      | ☐     | ☐             |       |
| 10    | ☐      | ☐      | ☐     | ☐             |       |
| 11    | ☐      | ☐      | ☐     | ☐             |       |
| 12    | ☐      | ☐      | ☐     | ☐             |       |
| 13    | ☐      | ☐      | ☐     | ☐             |       |
| 14    | ☐      | ☐      | ☐     | ☐             |       |
| 15    | ☐      | ☐      | ☐     | ☐             |       |
| 16    | ☐      | ☐      | ☐     | ☐             |       |
| 17    | ☐      | ☐      | ☐     | ☐             |       |
| 18    | ☐      | ☐      | ☐     | ☐             |       |
| 19    | ☐      | ☐      | ☐     | ☐             |       |
| 20    | ☐      | ☐      | ☐     | ☐             |       |

**Inhalation Rate**: _____/20
**Exhalation Rate**: _____/20
**Smoke Rate**: _____/20
**False Trigger Rate**: _____/20

---

### Stress Test (5 min)
- [ ] Completed without crash
- [ ] FPS stable (no >10% drop)
- [ ] Memory stable (no growth)
- [ ] Particles cleaned up properly
- [ ] No state machine errors
- [ ] Max particles: _____

---

## Performance Benchmarks

### FPS Test (30s)
- Average FPS: _____
- Minimum FPS: _____
- Maximum FPS: _____
- Meets 30 FPS target: ☐ Yes / ☐ No

### Latency
- Inhale pattern → INHALING state: _____ frames
- INHALING state → Glow visible: _____ frames
- Exhale pattern → EXHALING state: _____ frames
- EXHALING state → Smoke spawn: _____ frames

---

## Tracking Recovery
| Scenario | Tested | Recovered | Notes |
|----------|--------|-----------|-------|
| Face leaves frame | ☐ | ☐ Yes / ☐ No | |
| Hand leaves frame | ☐ | ☐ Yes / ☐ No | |
| Both leave | ☐ | ☐ Yes / ☐ No | |
| Returns after 5s | ☐ | ☐ Yes / ☐ No | |
| Returns after 30s | ☐ | ☐ Yes / ☐ No | |

---

## Debug Mode Verification
- [ ] Face landmarks visible
- [ ] Hand landmarks visible
- [ ] Distance line (cigarette ↔ mouth)
- [ ] State labels correct
- [ ] FPS counter working
- [ ] All toggle keys work (d, c, i, s, g, k, D)

---

## Clean Startup/Shutdown
- [ ] Application starts without errors
- [ ] Camera initializes
- [ ] MediaPipe loads
- [ ] Assets load
- [ ] Graceful exit on 'q'/'ESC'
- [ ] Camera released
- [ ] Windows closed
- [ ] No zombie processes

---

## Demo Mode (Production)
- [ ] `python src/main.py` (no debug args)
- [ ] No technical overlay visible
- [ ] Only AR effects visible
- [ ] 'D' key enables debug

---

## Final Metrics Calculation

### Detection Metrics (from repeated cycles)
- **True Positives (TP)**: _____
- **False Positives (FP)**: _____
- **False Negatives (FN)**: _____
- **True Negatives (TN)**: _____
- **Precision**: _____
- **Recall**: _____
- **F1 Score**: _____
- **Accuracy**: _____
- **False Positive Rate**: _____
- **False Negative Rate**: _____

---

## Overall Assessment

### Critical Criteria (Must Pass)
- [ ] No crashes during testing
- [ ] No permanent state corruption
- [ ] Camera releases properly
- [ ] Clean shutdown every time
- [ ] Core cycle works (inhale → glow → exhale → smoke)

### Quality Criteria
- [ ] FPS ≥ 30 average
- [ ] False positive rate < 10%
- [ ] False negative rate < 20%
- [ ] Cycle repeatability > 80%
- [ ] Stress test passes (5 min)

### Demo Readiness
- [ ] All unit tests pass
- [ ] Core functionality verified
- [ ] False positives acceptable
- [ ] Performance acceptable
- [ ] README complete
- [ ] Limitations documented

---

## Final Decision
**DEMO READY**: ☐ YES / ☐ NO

**If NO, blocking issues**:
1. _______________________________
2. _______________________________
3. _______________________________

**Sign-off**: ___________
**Date**: ___________