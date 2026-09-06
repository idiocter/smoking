import numpy as np
from utils.smoothing import Smoother


class SmokingState:
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    NEAR_MOUTH = "NEAR_MOUTH"
    INHALATION_CANDIDATE = "INHALATION_CANDIDATE"
    INHALING = "INHALING"
    COMPLETED = "COMPLETED"


class SmokingDetector:
    def __init__(self):
        self.NEAR_MOUTH_THRESHOLD = 80
        self.APPROACHING_FRAME_COUNT = 3
        self.NEAR_MOUTH_FRAME_COUNT = 3
        self.INHALATION_WINDOW = 10
        self.INHALATION_FRAME_COUNT = 4
        self.AWAY_FRAME_COUNT = 3
        self.MOUTH_OPENING_CHANGE_THRESHOLD = 4
        self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD = 0.3

        self._state = SmokingState.IDLE
        self._frames_in_state = 0
        self._prev_distance = None
        self._approach_count = 0
        self._near_mouth_count = 0
        self._inhalation_pattern_count = 0
        self._away_count = 0

        self._mouth_opening_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._mouth_aspect_ratio_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._prev_mouth_opening = None
        self._prev_mouth_aspect_ratio = None

        self.current_state = SmokingState.IDLE
        self.pattern_detected = False

    def update(self, cigarette_tracker, cigarette_mouth_detector, face_tracker):
        cig_state = cigarette_mouth_detector.get_state()
        distance = cigarette_mouth_detector.get_distance()

        mouth_data = face_tracker.get_mouth_measurements()
        mouth_opening = mouth_data.get('opening', 0)
        mouth_aspect_ratio = mouth_data.get('aspect_ratio', 0)

        self._mouth_opening_history.add(mouth_opening)
        self._mouth_aspect_ratio_history.add(mouth_aspect_ratio)

        self._update_state(cig_state, distance, mouth_opening, mouth_aspect_ratio)

        return self.current_state

    def _update_state(self, cig_state, distance, mouth_opening, mouth_aspect_ratio):
        if self._state == SmokingState.IDLE:
            self._frames_in_state = 0
            self._approach_count = 0
            self._near_mouth_count = 0
            self._inhalation_pattern_count = 0
            self._away_count = 0
            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio
            self.pattern_detected = False

            if cig_state == "APPROACHING":
                self._approach_count += 1
                if self._approach_count >= self.APPROACHING_FRAME_COUNT:
                    self._state = SmokingState.APPROACHING
                    self._frames_in_state = 0
                    self._approach_count = 0

        elif self._state == SmokingState.APPROACHING:
            self._frames_in_state += 1

            if cig_state == "NEAR":
                self._near_mouth_count += 1
                if self._near_mouth_count >= self.NEAR_MOUTH_FRAME_COUNT:
                    self._state = SmokingState.NEAR_MOUTH
                    self._frames_in_state = 0
                    self._near_mouth_count = 0
                    self._prev_mouth_opening = mouth_opening
                    self._prev_mouth_aspect_ratio = mouth_aspect_ratio
            elif cig_state == "FAR" or cig_state == "MOVING_AWAY":
                self._state = SmokingState.IDLE
            else:
                self._near_mouth_count = 0

        elif self._state == SmokingState.NEAR_MOUTH:
            self._frames_in_state += 1

            if cig_state != "NEAR":
                self._state = SmokingState.IDLE
                return

            opening_delta = abs(mouth_opening - self._prev_mouth_opening) if self._prev_mouth_opening is not None else 0
            ar_delta = abs(mouth_aspect_ratio - self._prev_mouth_aspect_ratio) if self._prev_mouth_aspect_ratio is not None else 0

            pattern_change = (
                opening_delta > self.MOUTH_OPENING_CHANGE_THRESHOLD or
                ar_delta > self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD
            )

            if pattern_change:
                self._inhalation_pattern_count += 1
                if self._inhalation_pattern_count >= self.INHALATION_FRAME_COUNT:
                    self._state = SmokingState.INHALATION_CANDIDATE
                    self._frames_in_state = 0
                    self._inhalation_pattern_count = 0
            else:
                self._inhalation_pattern_count = max(0, self._inhalation_pattern_count - 1)

            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio

        elif self._state == SmokingState.INHALATION_CANDIDATE:
            self._frames_in_state += 1

            if cig_state != "NEAR":
                self._state = SmokingState.IDLE
                return

            opening_delta = abs(mouth_opening - self._prev_mouth_opening) if self._prev_mouth_opening is not None else 0
            ar_delta = abs(mouth_aspect_ratio - self._prev_mouth_aspect_ratio) if self._prev_mouth_aspect_ratio is not None else 0

            pattern_continues = (
                opening_delta > self.MOUTH_OPENING_CHANGE_THRESHOLD or
                ar_delta > self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD
            )

            if pattern_continues:
                self._inhalation_pattern_count += 1
                if self._inhalation_pattern_count >= self.INHALATION_FRAME_COUNT:
                    self._state = SmokingState.INHALING
                    self._frames_in_state = 0
                    self.pattern_detected = True
                    self._inhalation_pattern_count = 0
            else:
                self._state = SmokingState.NEAR_MOUTH
                self._frames_in_state = 0
                self._inhalation_pattern_count = 0

            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio

        elif self._state == SmokingState.INHALING:
            self._frames_in_state += 1

            if cig_state == "MOVING_AWAY":
                self._away_count += 1
                if self._away_count >= self.AWAY_FRAME_COUNT:
                    self._state = SmokingState.COMPLETED
                    self._frames_in_state = 0
                    self._away_count = 0
            elif cig_state == "FAR":
                self._state = SmokingState.COMPLETED
                self._frames_in_state = 0
                self._away_count = 0
            else:
                self._away_count = 0

        elif self._state == SmokingState.COMPLETED:
            self._frames_in_state += 1
            if self._frames_in_state >= 2:
                self._state = SmokingState.IDLE

        self.current_state = self._state

    def get_state(self):
        return self.current_state

    def is_pattern_detected(self):
        return self.pattern_detected

    def get_debug_info(self):
        return {
            'state': self.current_state,
            'pattern_detected': self.pattern_detected,
            'frames_in_state': self._frames_in_state,
            'thresholds': {
                'near_mouth': self.NEAR_MOUTH_THRESHOLD,
                'mouth_opening_change': self.MOUTH_OPENING_CHANGE_THRESHOLD,
                'mouth_ar_change': self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD,
            },
            'frame_counts': {
                'approaching': self.APPROACHING_FRAME_COUNT,
                'near_mouth': self.NEAR_MOUTH_FRAME_COUNT,
                'inhalation': self.INHALATION_FRAME_COUNT,
                'away': self.AWAY_FRAME_COUNT,
            }
        }

    def reset(self):
        self._state = SmokingState.IDLE
        self._frames_in_state = 0
        self._approach_count = 0
        self._near_mouth_count = 0
        self._inhalation_pattern_count = 0
        self._away_count = 0
        self._mouth_opening_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._mouth_aspect_ratio_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._prev_mouth_opening = None
        self._prev_mouth_aspect_ratio = None
        self.current_state = SmokingState.IDLE
        self.pattern_detected = False