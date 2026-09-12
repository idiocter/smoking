#!/usr/bin/env python3
"""
Confusion matrix evaluation for smoking detection.
Run after manual testing to calculate metrics.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tests.evaluation.metrics import calculate_metrics, print_confusion_matrix


def evaluate_detection(results):
    """
    Calculate confusion matrix from manual test results.

    Args:
        results: Dict with keys:
            - tp: True positives (correctly detected smoking events)
            - fp: False positives (incorrectly detected)
            - fn: False negatives (missed events)
            - tn: True negatives (correctly rejected)
    """
    tp = results.get('tp', 0)
    fp = results.get('fp', 0)
    fn = results.get('fn', 0)
    tn = results.get('tn', 0)

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX EVALUATION")
    print("=" * 50)
    print(f"True Positives:  {tp}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Negatives:  {tn}")

    metrics = print_confusion_matrix(tp, fp, fn, tn)

    return metrics


def evaluate_per_event(results):
    """
    Evaluate at event level (inhalation, exhalation, smoke).

    Args:
        results: Dict with keys for each event type:
            - 'inhalation': {'expected': int, 'detected': int}
            - 'exhalation': {'expected': int, 'detected': int}
            - 'smoke': {'expected': int, 'detected': int}
    """
    print("\n" + "=" * 50)
    print("EVENT-LEVEL EVALUATION")
    print("=" * 50)

    for event_type in ['inhalation', 'exhalation', 'smoke']:
        if event_type in results:
            data = results[event_type]
            expected = data.get('expected', 0)
            detected = data.get('detected', 0)

            # Simple event-level accuracy
            if expected > 0:
                recall = detected / expected
                print(f"\n{event_type.capitalize()}:")
                print(f"  Expected: {expected}")
                print(f"  Detected: {detected}")
                print(f"  Recall:   {recall:.1%}")
            else:
                print(f"\n{event_type.capitalize()}: No expected events")


if __name__ == '__main__':
    # Example usage - replace with actual test results
    print("Confusion Matrix Evaluation Tool")
    print("Import this module and call evaluate_detection() with your results")
    print("\nExample:")
    print("  from tests.evaluate_confusion import evaluate_detection")
    print("  evaluate_detection({'tp': 18, 'fp': 2, 'fn': 3, 'tn': 45})")