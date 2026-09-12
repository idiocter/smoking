#!/usr/bin/env python3
"""
Evaluation logging and metrics for Virtual Smoking project.
"""

import csv
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class EventRecord:
    timestamp: float
    frame_number: int
    fps: float
    face_detected: bool
    hand_detected: bool
    cigarette_held: bool
    mouth_distance: Optional[float]
    cig_state: str
    smoking_state: str
    pattern_detected: bool
    exhalation_detected: bool
    glow_intensity: float
    smoke_active: bool
    particle_count: int


class EvaluationLogger:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.events: List[EventRecord] = []
        self.frame_number = 0
        self.start_time = time.time()

    def log_frame(self, data: Dict):
        """Log a single frame's evaluation data."""
        event = EventRecord(
            timestamp=time.time() - self.start_time,
            frame_number=self.frame_number,
            **data
        )
        self.events.append(event)
        self.frame_number += 1

    def save_csv(self, filename: str = None):
        """Save events to CSV."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', newline='') as f:
            if self.events:
                writer = csv.DictWriter(f, fieldnames=asdict(self.events[0]).keys())
                writer.writeheader()
                for event in self.events:
                    writer.writerow(asdict(event))
        print(f"Saved {len(self.events)} events to {filepath}")
        return filepath

    def save_json(self, filename: str = None):
        """Save events to JSON."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump([asdict(e) for e in self.events], f, indent=2)
        print(f"Saved {len(self.events)} events to {filepath}")
        return filepath

    def get_summary_stats(self) -> Dict:
        """Calculate summary statistics from logged events."""
        if not self.events:
            return {}

        total_frames = len(self.events)
        face_detected = sum(1 for e in self.events if e.face_detected)
        hand_detected = sum(1 for e in self.events if e.hand_detected)
        cigarette_held = sum(1 for e in self.events if e.cigarette_held)

        # State transitions
        smoking_states = [e.smoking_state for e in self.events]
        state_counts = {}
        for state in smoking_states:
            state_counts[state] = state_counts.get(state, 0) + 1

        # Event counts
        inhalation_events = sum(1 for e in self.events if e.pattern_detected)
        exhalation_events = sum(1 for e in self.events if e.exhalation_detected)
        smoke_events = sum(1 for e in self.events if e.smoke_active)

        # FPS stats
        fps_values = [e.fps for e in self.events if e.fps > 0]
        avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0
        min_fps = min(fps_values) if fps_values else 0
        max_fps = max(fps_values) if fps_values else 0

        return {
            'total_frames': total_frames,
            'duration_seconds': self.events[-1].timestamp if self.events else 0,
            'face_detection_rate': face_detected / total_frames if total_frames else 0,
            'hand_detection_rate': hand_detected / total_frames if total_frames else 0,
            'cigarette_held_rate': cigarette_held / total_frames if total_frames else 0,
            'state_distribution': state_counts,
            'inhalation_events': inhalation_events,
            'exhalation_events': exhalation_events,
            'smoke_events': smoke_events,
            'avg_fps': avg_fps,
            'min_fps': min_fps,
            'max_fps': max_fps,
        }

    def clear(self):
        self.events.clear()
        self.frame_number = 0
        self.start_time = time.time()


def calculate_metrics(tp: int, fp: int, fn: int, tn: int) -> Dict:
    """Calculate precision, recall, F1, accuracy."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0
    return {
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'false_positive_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        'false_negative_rate': fn / (fn + tp) if (fn + tp) > 0 else 0,
    }


def print_confusion_matrix(tp: int, fp: int, fn: int, tn: int):
    """Print confusion matrix."""
    metrics = calculate_metrics(tp, fp, fn, tn)
    print("\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Positive  Negative")
    print(f"Actual Positive    {tp:4d}     {fn:4d}")
    print(f"Actual Negative    {fp:4d}     {tn:4d}")
    print(f"\nPrecision:  {metrics['precision']:.3f}")
    print(f"Recall:     {metrics['recall']:.3f}")
    print(f"F1 Score:   {metrics['f1']:.3f}")
    print(f"Accuracy:   {metrics['accuracy']:.3f}")
    print(f"FP Rate:    {metrics['false_positive_rate']:.3f}")
    print(f"FN Rate:    {metrics['false_negative_rate']:.3f}")
    return metrics