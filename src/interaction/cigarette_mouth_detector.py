import numpy as np
from utils.geometry import distance
from utils.smoothing import Smoother


class CigaretteMouthState:
    FAR = "FAR"
    APPROACHING = "APPROACHING"
    NEAR = "NEAR"
    MOVING_AWAY = "MOVING_AWAY"


class CigaretteMouthDetector:
    def __init__(self,
                 near_threshold=80,
                 approach_frames=3,
                 near_frames=3,
                 away_frames=3,
                 distance_smoothing=3):
        self.near_threshold = near_threshold
        self.approach_frames = approach_frames
        self.near_frames = near_frames
        self.away_frames = away_frames

        self._distance_history = Smoother(window_size=distance_smoothing)
        self._state = CigaretteMouthState.FAR
        self._frames_in_state = 0
        self._prev_distance = None
        self._approach_count = 0
        self._near_count = 0
        self._away_count = 0

        self.current_distance = None
        self.current_state = CigaretteMouthState.FAR

    def _calculate_distance(self, cig_pos, mouth_center):
        if cig_pos is None or mouth_center is None:
            return None
        return distance(cig_pos, mouth_center)

    def update(self, cigarette_tracker, face_tracker):
        mouth_center = face_tracker.get_mouth_center()
        cig_pos = None

        if cigarette_tracker.is_held:
            cig_pos = cigarette_tracker.get_mouth_end_position(mouth_center)

        raw_distance = self._calculate_distance(cig_pos, mouth_center)
        self.current_distance = self._distance_history.add(raw_distance) if raw_distance is not None else None

        if self.current_distance is None:
            self._reset_state()
            return self.current_state

        self._update_state()
        return self.current_state

    def _reset_state(self):
        self._state = CigaretteMouthState.FAR
        self._frames_in_state = 0
        self._prev_distance = None
        self._approach_count = 0
        self._near_count = 0
        self._away_count = 0

    def _update_state(self):
        dist = self.current_distance

        if self._prev_distance is not None:
            delta = dist - self._prev_distance

            if self._state == CigaretteMouthState.FAR:
                if dist < self.near_threshold:
                    self._near_count += 1
                    self._approach_count = 0
                    self._away_count = 0
                    if self._near_count >= self.near_frames:
                        self._state = CigaretteMouthState.NEAR
                        self._frames_in_state = 0
                        self._near_count = 0
                elif delta < -2:
                    self._approach_count += 1
                    self._near_count = 0
                    self._away_count = 0
                    if self._approach_count >= self.approach_frames:
                        self._state = CigaretteMouthState.APPROACHING
                        self._frames_in_state = 0
                        self._approach_count = 0
                else:
                    self._approach_count = max(0, self._approach_count - 1)
                    self._near_count = 0
                    self._away_count = 0

            elif self._state == CigaretteMouthState.APPROACHING:
                self._frames_in_state += 1
                if dist < self.near_threshold:
                    self._near_count += 1
                    self._away_count = 0
                    if self._near_count >= self.near_frames:
                        self._state = CigaretteMouthState.NEAR
                        self._frames_in_state = 0
                        self._near_count = 0
                elif delta > 2:
                    self._away_count += 1
                    self._near_count = 0
                    if self._away_count >= self.away_frames:
                        self._state = CigaretteMouthState.MOVING_AWAY
                        self._frames_in_state = 0
                        self._away_count = 0
                else:
                    self._away_count = 0

            elif self._state == CigaretteMouthState.NEAR:
                self._frames_in_state += 1
                if dist >= self.near_threshold * 1.2:
                    self._away_count += 1
                    self._near_count = 0
                    if self._away_count >= self.away_frames:
                        self._state = CigaretteMouthState.MOVING_AWAY
                        self._frames_in_state = 0
                        self._away_count = 0
                else:
                    self._away_count = 0
                    self._near_count = min(self.near_frames, self._near_count + 1)

            elif self._state == CigaretteMouthState.MOVING_AWAY:
                self._frames_in_state += 1
                if dist < self.near_threshold:
                    self._near_count += 1
                    self._away_count = 0
                    if self._near_count >= self.near_frames:
                        self._state = CigaretteMouthState.NEAR
                        self._frames_in_state = 0
                        self._near_count = 0
                elif dist > self.near_threshold * 2:
                    self._state = CigaretteMouthState.FAR
                    self._frames_in_state = 0
                    self._away_count = 0
                else:
                    self._near_count = 0

        self._prev_distance = dist
        self.current_state = self._state

    def get_state(self):
        return self.current_state

    def get_distance(self):
        return self.current_distance

    def get_debug_info(self):
        return {
            'state': self.current_state,
            'distance': self.current_distance,
            'threshold': self.near_threshold,
            'frames_in_state': self._frames_in_state,
        }