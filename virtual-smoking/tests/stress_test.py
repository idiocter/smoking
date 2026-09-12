#!/usr/bin/env python3
"""
Stress test for Virtual Smoking - runs extended session to check stability.
"""

import sys
import os
import time
import tracemalloc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from camera.camera import Camera
from vision.face_tracker import FaceTracker
from vision.hand_tracker import HandTracker
from interaction.cigarette_tracker import CigaretteTracker
from interaction.cigarette_mouth_detector import CigaretteMouthDetector
from interaction.smoking_detector import SmokingDetector
from effects.cigarette import CigaretteRenderer, CigaretteRendererFallback
from effects.glow import GlowEffect
from effects.smoke import SmokeEffect
from config import Config


def run_stress_test(duration_minutes=5):
    """Run stress test for specified minutes."""

    camera = Camera(
        device_index=Config.CAMERA['device_index'],
        width=Config.CAMERA['width'],
        height=Config.CAMERA['height']
    )
    face_tracker = FaceTracker(
        max_faces=Config.FACE_TRACKER['max_faces'],
        min_detection_confidence=Config.FACE_TRACKER['min_detection_confidence'],
        min_tracking_confidence=Config.FACE_TRACKER['min_tracking_confidence']
    )
    hand_tracker = HandTracker(
        max_hands=Config.HAND_TRACKER['max_hands'],
        min_detection_confidence=Config.HAND_TRACKER['min_detection_confidence'],
        min_tracking_confidence=Config.HAND_TRACKER['min_tracking_confidence']
    )
    cigarette_tracker = CigaretteTracker()
    cigarette_mouth_detector = CigaretteMouthDetector()
    smoking_detector = SmokingDetector()
    cigarette_renderer = CigaretteRenderer()
    glow_effect = GlowEffect(
        max_intensity=Config.GLOW_EFFECT['max_intensity'],
        fade_in_speed=Config.GLOW_EFFECT['fade_in_speed'],
        fade_out_speed=Config.GLOW_EFFECT['fade_out_speed']
    )
    smoke_effect = SmokeEffect()
    fallback_renderer = CigaretteRendererFallback()

    try:
        camera.open()
    except RuntimeError as e:
        print(f"Error: {e}")
        return False

    print(f"Running stress test for {duration_minutes} minutes...")

    tracemalloc.start()

    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    frame_count = 0
    fps_samples = []

    max_particles = 0
    state_errors = 0

    try:
        while time.time() < end_time:
            frame_start = time.time()

            frame = camera.read()
            if frame is None:
                continue

            face_detected = face_tracker.process(frame)
            hand_detected = hand_tracker.process(frame)

            hand_landmarks = hand_tracker.get_all_landmarks()
            cigarette_tracker.update(hand_landmarks)

            mouth_center = None
            if face_detected and cigarette_tracker.is_held:
                mouth_center = face_tracker.get_mouth_center()
                cigarette_mouth_detector.update(cigarette_tracker, face_tracker)
                smoking_state = smoking_detector.update(
                    cigarette_tracker, cigarette_mouth_detector, face_tracker
                )

                # Check for invalid state transitions
                if smoking_state not in [
                    "IDLE", "APPROACHING", "NEAR_MOUTH", "INHALATION_CANDIDATE",
                    "INHALING", "EXHALATION_CANDIDATE", "EXHALING", "COMPLETED"
                ]:
                    state_errors += 1
                    print(f"WARNING: Invalid state: {smoking_state}")

            is_inhaling = (smoking_detector.get_state() == "INHALING")
            glow_effect.set_target(is_inhaling)
            glow_effect.update()

            exhale_detected = smoking_detector.is_exhalation_detected()
            smoke_effect.update(exhale_detected, mouth_center)

            max_particles = max(max_particles, smoke_effect.get_particle_count())

            frame_time = time.time() - frame_start
            if frame_time > 0:
                fps_samples.append(1.0 / frame_time)

            frame_count += 1

            if frame_count % 300 == 0:  # Every ~10 seconds
                current, peak = tracemalloc.get_traced_memory()
                elapsed = time.time() - start_time
                avg_fps = sum(fps_samples[-30:]) / min(30, len(fps_samples))
                print(f"  {elapsed:.0f}s: {frame_count} frames, {avg_fps:.1f} FPS, "
                      f"Mem: {current/1024/1024:.1f}MB, "
                      f"Particles: {smoke_effect.get_particle_count()}, "
                      f"State errors: {state_errors}")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        camera.close()
        face_tracker.close()
        hand_tracker.close()
        tracemalloc.stop()

    # Final stats
    total_time = time.time() - start_time
    avg_fps = sum(fps_samples) / len(fps_samples) if fps_samples else 0
    min_fps = min(fps_samples) if fps_samples else 0
    max_fps = max(fps_samples) if fps_samples else 0

    current, peak = tracemalloc.get_traced_memory()

    print("\n" + "=" * 50)
    print("STRESS TEST RESULTS")
    print("=" * 50)
    print(f"Duration:        {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Total Frames:    {frame_count}")
    print(f"Avg FPS:         {avg_fps:.1f}")
    print(f"Min FPS:         {min_fps:.1f}")
    print(f"Max FPS:         {max_fps:.1f}")
    print(f"State Errors:    {state_errors}")
    print(f"Max Particles:   {max_particles}")
    print(f"Memory (current): {current/1024/1024:.1f}MB")
    print(f"Memory (peak):    {peak/1024/1024:.1f}MB")
    print("=" * 50)

    results = {
        'duration_seconds': total_time,
        'total_frames': frame_count,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'state_errors': state_errors,
        'max_particles': max_particles,
        'memory_peak_mb': peak/1024/1024,
    }

    return results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Stress test for Virtual Smoking')
    parser.add_argument('--duration', type=int, default=5, help='Duration in minutes')
    args = parser.parse_args()

    results = run_stress_test(args.duration)

    # Save results
    import json
    with open('results/stress_test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/stress_test_results.json")