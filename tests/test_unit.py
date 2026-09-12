#!/usr/bin/env python3
"""
Unit tests for Virtual Smoking components (no webcam required).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils.geometry import distance, midpoint, angle_between, vector_angle, normalize, clamp
from utils.smoothing import Smoother, OneEuroFilter, OneEuroFilter2D, AngleOneEuroFilter
from interaction.cigarette_tracker import CigaretteTracker
from interaction.cigarette_mouth_detector import CigaretteMouthDetector, CigaretteMouthState
from interaction.smoking_detector import SmokingDetector, SmokingState
from effects.smoke import SmokeEffect, SmokeParticle


def create_mock_hand_landmarks(thumb_tip=(100, 100), index_tip=(120, 100),
                               index_mcp=(120, 80), middle_tip=(140, 100),
                               middle_mcp=(140, 80), thumb_ip=(110, 100)):
    """Create mock hand landmarks dict."""
    return {
        'thumb_tip': thumb_tip,
        'thumb_ip': thumb_ip,
        'index_tip': index_tip,
        'index_mcp': index_mcp,
        'middle_tip': middle_tip,
        'middle_mcp': middle_mcp,
    }


def create_mock_face_measurements(center=(320, 240), opening=10, width=50, height=15, aspect_ratio=3.33):
    """Create mock face measurements dict."""
    return {
        'center': center,
        'opening': opening,
        'width': width,
        'height': height,
        'aspect_ratio': aspect_ratio,
    }


class MockFaceTracker:
    def __init__(self, measurements=None):
        self.measurements = measurements or create_mock_face_measurements()

    def get_mouth_center(self):
        return self.measurements['center']

    def get_mouth_measurements(self):
        return self.measurements

    def is_detected(self):
        return True


class MockHandTracker:
    def __init__(self, landmarks=None):
        self.landmarks = landmarks or create_mock_hand_landmarks()

    def get_all_landmarks(self):
        return self.landmarks


class MockCigaretteTracker:
    def __init__(self, held=True, position=(100, 100), rotation=0.0):
        self.is_held = held
        self.position = position
        self.rotation = rotation
        self.length = 140

    def get_mouth_end_position(self, mouth_center):
        if self.position is None:
            return None
        import math
        dx = math.cos(self.rotation) * (self.length / 2)
        dy = math.sin(self.rotation) * (self.length / 2)
        return (self.position[0] + dx, self.position[1] + dy)


# ==================== GEOMETRY TESTS ====================

def test_geometry():
    print("Testing geometry functions...")
    # Distance
    assert distance((0, 0), (3, 4)) == 5.0
    assert distance((1, 1), (4, 5)) == 5.0

    # Midpoint
    assert midpoint((0, 0), (10, 10)) == (5.0, 5.0)
    assert midpoint((-5, 5), (5, -5)) == (0.0, 0.0)

    # Vector angle
    assert abs(vector_angle((1, 0)) - 0) < 0.001
    assert abs(vector_angle((0, 1)) - np.pi/2) < 0.001
    assert abs(vector_angle((-1, 0)) - np.pi) < 0.001
    assert abs(vector_angle((0, -1)) + np.pi/2) < 0.001

    # Angle between
    angle = angle_between((1, 0), (0, 0), (0, 1))
    assert abs(angle - np.pi/2) < 0.001

    # Normalize
    assert normalize((3, 4)) == (0.6, 0.8)
    assert normalize((0, 0)) == (0.0, 0.0)

    # Clamp
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10

    print("  Geometry tests PASSED")


# ==================== SMOOTHING TESTS ====================

def test_smoothing():
    print("Testing smoothing classes...")

    # Smoother (moving average)
    s = Smoother(window_size=3)
    s.add((0, 0))
    s.add((10, 10))
    s.add((20, 20))
    assert s.get() == (10.0, 10.0)
    s.add((30, 30))
    assert s.get() == (20.0, 20.0)
    s.clear()
    assert s.get() is None

    # OneEuroFilter (scalar)
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0)
    assert f(10.0) == 10.0
    result = f(12.0)
    assert 10.0 < result < 12.0
    f.reset()
    assert f(5.0) == 5.0

    # OneEuroFilter2D (position)
    f2d = OneEuroFilter2D(freq=30.0, mincutoff=1.5, beta=0.3)
    result = f2d((100, 100))
    assert result == (100, 100)
    result = f2d((110, 110))
    assert 100 < result[0] < 110
    assert 100 < result[1] < 110
    f2d.reset()
    assert f2d((50, 50)) == (50, 50)

    # AngleOneEuroFilter
    fa = AngleOneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0)
    assert fa(0.0) == 0.0
    result = fa(0.1)
    assert 0.0 < result < 0.1

    # Test angle wrapping (179° -> -179° should be small change)
    fa2 = AngleOneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0)
    fa2(3.1)  # ~179 deg
    result = fa2(-3.1)  # ~-179 deg
    assert result > 0  # Should not wrap the wrong way

    fa2.reset()
    assert fa2(1.0) == 1.0

    print("  Smoothing tests PASSED")


# ==================== CIGARETTE TRACKER TESTS ====================

def test_cigarette_tracker():
    print("Testing cigarette tracker...")

    tracker = CigaretteTracker()
    hand = MockHandTracker()

    # Test with valid landmarks
    tracker.update(hand.landmarks)
    assert tracker.is_held
    assert tracker.position is not None
    assert tracker.rotation is not None

    # Test mouth end position
    mouth_center = (320, 240)
    end_pos = tracker.get_mouth_end_position(mouth_center)
    assert end_pos is not None

    # Test with no hand
    tracker.update(None)
    # Should still be held for a few frames
    assert tracker.is_held or tracker._frames_lost > 0

    # Test after many lost frames
    for _ in range(15):
        tracker.update(None)
    assert not tracker.is_held

    # Test reset
    tracker.update(hand.landmarks)
    tracker.reset()
    assert not tracker.is_held
    assert tracker.position is None

    print("  Cigarette tracker tests PASSED")


# ==================== CIGARETTE-MOUTH DETECTOR TESTS ====================

def test_cigarette_mouth_detector():
    print("Testing cigarette-mouth detector...")

    detector = CigaretteMouthDetector()
    # Cigarette positioned so tip is at mouth center (accounting for length/2 offset)
    cig = MockCigaretteTracker(held=True, position=(250, 240), rotation=0.0)  # tip at 320, 240
    face = MockFaceTracker(create_mock_face_measurements(center=(320, 240)))

    # Cigarette at mouth center -> distance 0 -> NEAR (needs multiple frames)
    state = None
    for _ in range(5):
        state = detector.update(cig, face)
    print(f"  Cig pos: {cig.position}, State: {state}, Dist: {detector.get_distance()}")
    # Distance is from tip to mouth center, tip is at (320, 240), mouth at (320, 240) -> dist 0
    assert state == CigaretteMouthState.NEAR or state == CigaretteMouthState.APPROACHING, f"Got {state}"

    # Move cigarette far
    cig.position = (500, 500)
    for _ in range(5):
        state = detector.update(cig, face)
    # Should eventually become FAR or MOVING_AWAY

    # Test distance
    dist = detector.get_distance()
    assert dist is not None

    # Test reset
    detector._reset_state()
    assert detector.current_state == CigaretteMouthState.FAR

    print("  Cigarette-mouth detector tests PASSED")


# ==================== SMOKING DETECTOR TESTS ====================

def test_smoking_detector():
    print("Testing smoking detector...")

    detector = SmokingDetector()
    cig = MockCigaretteTracker(held=True)
    cig_mouth = CigaretteMouthDetector()
    face = MockFaceTracker()

    # Test initial state
    state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.IDLE

    # Test valid inhalation sequence (simplified)
    # This is complex to test fully without real data

    # Test reset
    detector.reset()
    assert detector.current_state == SmokingState.IDLE
    assert not detector.pattern_detected
    assert not detector.exhalation_detected

    # Test debug info
    debug = detector.get_debug_info()
    assert 'state' in debug
    assert 'thresholds' in debug
    assert 'frame_counts' in debug

    print("  Smoking detector tests PASSED")


# ==================== SMOKE EFFECT TESTS ====================

def test_smoke_effect():
    print("Testing smoke effect...")

    effect = SmokeEffect()
    mouth_center = (320, 240)

    # Test no exhalation
    effect.update(False, mouth_center)
    assert not effect.is_active()
    assert effect.get_particle_count() == 0

    # Test exhalation triggers particles
    effect.update(True, mouth_center)
    assert effect.is_active()
    assert effect.get_particle_count() > 0

    # Test continued exhalation doesn't spawn more
    initial_count = effect.get_particle_count()
    effect.update(True, mouth_center)
    # Should not double the particles (only triggers on rising edge)

    # Test no exhalation -> particles eventually expire
    for _ in range(100):
        effect.update(False, mouth_center)
    assert not effect.is_active() or effect.get_particle_count() < initial_count

    # Test reset
    effect.update(True, mouth_center)
    effect.reset()
    assert not effect.is_active()
    assert effect.get_particle_count() == 0

    print("  Smoke effect tests PASSED")


# ==================== STATE MACHINE INTEGRATION TEST ====================

def test_state_machine_transitions():
    print("Testing state machine transitions...")

    detector = SmokingDetector()
    cig = MockCigaretteTracker(held=True)
    cig_mouth = CigaretteMouthDetector()
    face = MockFaceTracker()

    # Test IDLE -> APPROACHING transition requires consecutive frames
    # This is hard to test without full pipeline, but verify states exist
    states = [
        SmokingState.IDLE,
        SmokingState.APPROACHING,
        SmokingState.NEAR_MOUTH,
        SmokingState.INHALATION_CANDIDATE,
        SmokingState.INHALING,
        SmokingState.EXHALATION_CANDIDATE,
        SmokingState.EXHALING,
        SmokingState.COMPLETED,
    ]

    for state in states:
        assert isinstance(state, str)

    print("  State machine tests PASSED")


# ==================== RUN ALL TESTS ====================

if __name__ == '__main__':
    print("=" * 50)
    print("VIRTUAL SMOKING - UNIT TEST SUITE")
    print("=" * 50)

    test_geometry()
    test_smoothing()
    test_cigarette_tracker()
    test_cigarette_mouth_detector()
    test_smoking_detector()
    test_smoke_effect()
    test_state_machine_transitions()

    print("=" * 50)
    print("ALL UNIT TESTS PASSED")
    print("=" * 50)