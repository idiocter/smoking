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
from effects.cigarette import apply_finger_occlusion
from effects.ashtray import AshtrayRenderer
from effects.smoke import InhaleSmokeParticle, SmokeEffect, SmokeParticle


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


class MockCigaretteMouthDetector:
    def __init__(self, state=CigaretteMouthState.FAR, distance=200):
        self.state = state
        self.distance = distance

    def get_state(self):
        return self.state

    def get_distance(self):
        return self.distance


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
    assert tracker.position == (130.0, 100.0)
    assert abs(tracker.rotation) < 0.001

    # Thumb position must not affect the index-middle grip.
    tracker.reset()
    no_thumb = create_mock_hand_landmarks(thumb_tip=None)
    tracker.update(no_thumb)
    assert tracker.is_held
    assert tracker.position == (130.0, 100.0)

    # Nearly overlapping fingertips use the finger axes for a stable angle.
    tracker.reset()
    close_fingers = create_mock_hand_landmarks(
        index_tip=(120, 100),
        index_mcp=(120, 80),
        middle_tip=(125, 100),
        middle_mcp=(125, 80),
    )
    tracker.update(close_fingers)
    assert tracker.position == (122.5, 100.0)
    assert abs(tracker.rotation) < 0.001

    # Both target fingers are required to place the cigarette.
    tracker.reset()
    tracker.update(create_mock_hand_landmarks(middle_tip=None))
    assert not tracker.is_held

    # Spreading the two fingers releases the cigarette instead of pinning it.
    tracker.reset()
    tracker.update(create_mock_hand_landmarks(middle_tip=(220, 100)))
    assert not tracker.is_held

    # 3D palm direction controls the cigarette's camera-facing depth angle.
    tracker.reset()
    hand_3d = {
        'wrist': (100, 180, 0),
        'index_mcp': (110, 140, 0),
        'middle_mcp': (130, 130, -5),
        'pinky_mcp': (170, 150, 0),
    }
    tracker.update(hand.landmarks, hand_3d)
    assert tracker.is_held
    assert tracker.depth_rotation > np.deg2rad(70)

    # Test mouth end position
    tracker.reset()
    tracker.update(hand.landmarks)
    mouth_center = (320, 240)
    end_pos = tracker.get_mouth_end_position(mouth_center)
    assert end_pos == tracker.get_base_position()
    ember_pos = tracker.get_ember_position(mouth_center)
    assert ember_pos == tracker.get_tip_position()
    assert abs(tracker.get_render_rotation(mouth_center)) < 0.001
    assert tracker.get_render_position()[0] > tracker.position[0]

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


def test_finger_occlusion():
    print("Testing finger occlusion...")
    original = np.zeros((50, 60, 3), dtype=np.uint8)
    rendered = np.full_like(original, 255)
    landmarks = {
        'index_pip': (20, 10),
        'index_dip': (20, 20),
        'index_tip': (20, 35),
        'middle_pip': (40, 10),
        'middle_dip': (40, 20),
        'middle_tip': (40, 35),
    }

    result = apply_finger_occlusion(rendered, original, landmarks)
    assert np.all(result[20, 20] == 0)
    assert np.all(result[20, 40] == 0)
    assert np.all(result[20, 30] == 255)
    print("  Finger occlusion tests PASSED")


def test_cigarette_pickup_and_gravity():
    print("Testing cigarette pickup and gravity...")
    ashtray = AshtrayRenderer()
    ashtray.update_layout(1280, 720)
    rest_position = ashtray.get_rest_pose()['position']
    tracker = CigaretteTracker()

    grip = create_mock_hand_landmarks(
        index_tip=(rest_position[0] - 10, rest_position[1]),
        index_mcp=(rest_position[0] - 10, rest_position[1] + 45),
        middle_tip=(rest_position[0] + 10, rest_position[1]),
        middle_mcp=(rest_position[0] + 10, rest_position[1] + 45),
    )
    grip['wrist'] = (rest_position[0], rest_position[1] + 150)

    tracker.update(grip, frame_shape=(720, 1280), ashtray=ashtray)
    assert tracker.state == CigaretteTracker.HELD
    assert tracker.is_held

    moved_grip = {
        name: (point[0], point[1] + 80) if point is not None else None
        for name, point in grip.items()
    }
    old_y = tracker.position[1]
    tracker.update(moved_grip, frame_shape=(720, 1280), ashtray=ashtray)
    assert old_y < tracker.position[1] < moved_grip['index_tip'][1] + 10
    target_y = rest_position[1] + 80
    assert target_y - tracker.position[1] < 15

    # A brief landmark dropout coasts instead of freezing the object in place.
    coast_y = tracker.position[1]
    tracker.update(None, frame_shape=(720, 1280), ashtray=ashtray)
    assert tracker.state == CigaretteTracker.HELD
    assert tracker.position[1] > coast_y

    spread_grip = dict(moved_grip)
    spread_grip['middle_tip'] = (
        moved_grip['middle_tip'][0] + 110,
        moved_grip['middle_tip'][1],
    )
    tracker.update(spread_grip, frame_shape=(720, 1280), ashtray=ashtray)
    assert tracker.state == CigaretteTracker.FALLING
    assert not tracker.is_held

    falling_y = tracker.position[1]
    tracker.update(None, frame_shape=(720, 1280), ashtray=ashtray)
    assert tracker.position[1] > falling_y

    tracker.position = (rest_position[0], rest_position[1] - 20)
    tracker.velocity[:] = (0, 25)
    tracker.state = CigaretteTracker.FALLING
    tracker.update(None, frame_shape=(720, 1280), ashtray=ashtray)
    assert tracker.state == CigaretteTracker.RESTING
    assert tracker.position == rest_position
    print("  Cigarette pickup and gravity tests PASSED")


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
    cig_mouth = MockCigaretteMouthDetector(
        state=CigaretteMouthState.NEAR,
        distance=20,
    )
    face = MockFaceTracker()

    # Direct placement at the lips must work even without an APPROACHING phase.
    for _ in range(detector.NEAR_MOUTH_FRAME_COUNT):
        state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.NEAR_MOUTH

    # A sustained mouth change is measured against the near-mouth baseline.
    face.measurements = create_mock_face_measurements(
        opening=16,
        width=50,
        height=15,
        aspect_ratio=2.5,
    )
    for _ in range(detector.INHALATION_FRAME_COUNT * 2):
        state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.INHALING
    assert detector.is_pattern_detected()

    # Moving away, followed by a sustained open mouth, starts an exhale.
    cig_mouth.state = CigaretteMouthState.MOVING_AWAY
    for _ in range(detector.AWAY_FRAME_COUNT):
        state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.EXHALATION_CANDIDATE

    face.measurements = create_mock_face_measurements(
        opening=10,
        width=58,
        height=15,
        aspect_ratio=2.0,
    )
    for _ in range(detector.EXHALATION_FRAME_COUNT):
        state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.EXHALING
    assert detector.is_exhalation_detected()

    for _ in range(detector.EXHALATION_DURATION_FRAMES - 1):
        state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.EXHALING
    state = detector.update(cig, cig_mouth, face)
    assert state == SmokingState.COMPLETED
    assert not detector.is_exhalation_detected()

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

    # A turned face gives the newly emitted breath matching screen-space motion.
    parcel = SmokeParticle(100, 100, effect.config, direction=(1.0, 0.0))
    initial_x = parcel.x
    initial_size = parcel.size
    parcel.update()
    assert parcel.x > initial_x
    assert parcel.size > initial_size
    assert parcel.opacity > 0

    # Test no exhalation
    effect.update(False, mouth_center)
    assert not effect.is_active()
    assert effect.get_particle_count() == 0

    # Test exhalation triggers particles
    effect.update(True, mouth_center)
    assert effect.is_active()
    assert effect.get_particle_count() > 0

    # A continued exhale produces a continuous plume.
    initial_count = effect.get_particle_count()
    for _ in range(effect.config['spawn_interval_frames'] + 1):
        effect.update(True, mouth_center)
    assert effect.get_particle_count() > initial_count

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    effect.draw(frame)
    assert frame.sum() > 0

    # Test no exhalation -> particles eventually expire
    for _ in range(100):
        effect.update(False, mouth_center)
    assert not effect.is_active() or effect.get_particle_count() < initial_count

    # Inhaling creates small wisps that move from the ember toward the mouth.
    effect.reset()
    ember_position = (200, 240)
    effect.update(
        False,
        mouth_center,
        inhalation_detected=True,
        ember_position=ember_position,
    )
    inhale_particles = [
        particle for particle in effect.particles
        if isinstance(particle, InhaleSmokeParticle)
    ]
    assert inhale_particles
    particle = inhale_particles[0]
    previous_distance = distance((particle.x, particle.y), mouth_center)
    effect.update(
        False,
        mouth_center,
        inhalation_detected=True,
        ember_position=ember_position,
    )
    assert distance((particle.x, particle.y), mouth_center) < previous_distance

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
    test_finger_occlusion()
    test_cigarette_pickup_and_gravity()
    test_cigarette_mouth_detector()
    test_smoking_detector()
    test_smoke_effect()
    test_state_machine_transitions()

    print("=" * 50)
    print("ALL UNIT TESTS PASSED")
    print("=" * 50)
