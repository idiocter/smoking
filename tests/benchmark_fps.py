#!/usr/bin/env python3
"""
FPS Benchmark script for Virtual Smoking.
Run with: python tests/benchmark_fps.py [--duration SECONDS] [--debug]
"""

import sys
import os
import argparse
import time
import json
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


def run_benchmark(duration=30, debug=False):
    """Run FPS benchmark for specified duration."""

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
        return None

    print(f"Running benchmark for {duration} seconds...")
    if debug:
        print("Debug mode enabled")

    fps_samples = []
    frame_times = []
    start_time = time.time()
    frame_count = 0

    while time.time() - start_time < duration:
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
            smoking_detector.update(cigarette_tracker, cigarette_mouth_detector, face_tracker)

        is_inhaling = (smoking_detector.get_state() == "INHALING")
        glow_effect.set_target(is_inhaling)
        glow_effect.update()
        smoke_effect.update(smoking_detector.is_exhalation_detected(), mouth_center)

        # Render (skip display for benchmark)
        if cigarette_tracker.is_held and cigarette_tracker.position is not None:
            pass  # Skip actual rendering for pure FPS measurement

        frame_time = time.time() - frame_start
        frame_times.append(frame_time)

        if frame_time > 0:
            fps = 1.0 / frame_time
            fps_samples.append(fps)

        frame_count += 1

        if frame_count % 30 == 0 and debug:
            print(f"Frame {frame_count}: {fps:.1f} FPS")

    total_time = time.time() - start_time

    camera.close()
    face_tracker.close()
    hand_tracker.close()

    # Calculate statistics
    if not fps_samples:
        print("No frames captured!")
        return None

    avg_fps = sum(fps_samples) / len(fps_samples)
    min_fps = min(fps_samples)
    max_fps = max(fps_samples)

    # Percentiles
    sorted_fps = sorted(fps_samples)
    p50 = sorted_fps[len(sorted_fps) // 2]
    p95 = sorted_fps[int(len(sorted_fps) * 0.95)]
    p99 = sorted_fps[int(len(sorted_fps) * 0.99)]

    results = {
        'duration_seconds': total_time,
        'total_frames': frame_count,
        'avg_fps': avg_fps,
        'min_fps': min_fps,
        'max_fps': max_fps,
        'median_fps': p50,
        'p95_fps': p95,
        'p99_fps': p99,
        'target_fps': Config.CAMERA['fps'],
        'meets_target': avg_fps >= Config.CAMERA['fps'],
        'fps_samples': len(fps_samples),
    }

    print("\n" + "=" * 50)
    print("FPS BENCHMARK RESULTS")
    print("=" * 50)
    print(f"Duration:        {total_time:.1f}s")
    print(f"Total Frames:    {frame_count}")
    print(f"Avg FPS:         {avg_fps:.1f}")
    print(f"Median FPS:      {p50:.1f}")
    print(f"Min FPS:         {min_fps:.1f}")
    print(f"Max FPS:         {max_fps:.1f}")
    print(f"95th Percentile: {p95:.1f}")
    print(f"99th Percentile: {p99:.1f}")
    print(f"Target FPS:      {Config.CAMERA['fps']}")
    print(f"Meets Target:    {'YES' if results['meets_target'] else 'NO'}")
    print("=" * 50)

    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FPS Benchmark for Virtual Smoking')
    parser.add_argument('--duration', type=int, default=30, help='Benchmark duration in seconds')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    args = parser.parse_args()

    results = run_benchmark(args.duration, args.debug)

    if results and args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")