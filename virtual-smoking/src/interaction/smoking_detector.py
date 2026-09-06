import numpy as np
from utils.smoothing import Smoother
from config import Config


class SmokingState:
    IDLE = "IDLE"
    APPROACHING = "APPROACHING"
    NEAR_MOUTH = "NEAR_MOUTH"
    INHALATION_CANDIDATE = "INHALATION_CANDIDATE"
    INHALING = "INHALING"
    EXHALATION_CANDIDATE = "EXHALATION_CANDIDATE"
    EXHALING = "EXHALING"
    COMPLETED = "COMPLETED"


class SmokingDetector:
    def __init__(self):
        cfg = Config.SMOKING_DETECTOR
        self.NEAR_MOUTH_THRESHOLD = cfg['near_mouth_threshold']
        self.APPROACHING_FRAME_COUNT = cfg['approaching_frame_count']
        self.NEAR_MOUTH_FRAME_COUNT = cfg['near_mouth_frame_count']
        self.INHALATION_WINDOW = cfg['inhalation_window']
        self.INHALATION_FRAME_COUNT = cfg['inhalation_frame_count']
        self.AWAY_FRAME_COUNT = cfg['away_frame_count']
        self.MOUTH_OPENING_CHANGE_THRESHOLD = cfg['mouth_opening_change_threshold']
        self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD = cfg['mouth_aspect_ratio_change_threshold']
        self.EXHALATION_WINDOW = cfg['exhalation_window']
        self.EXHALATION_FRAME_COUNT = cfg['exhalation_frame_count']
        self.EXHALATION_STABILITY_FRAMES = cfg['exhalation_stability_frames']
        self.EXHALATION_MOUTH_OPENING_THRESHOLD = cfg['exhalation_mouth_opening_threshold']
        self.EXHALATION_MOUTH_WIDTH_CHANGE_THRESHOLD = cfg['exhalation_mouth_width_change_threshold']
        self.AWAY_FROM_MOUTH_THRESHOLD = cfg['away_from_mouth_threshold']

        self._state = SmokingState.IDLE
        self._frames_in_state = 0
        self._prev_distance = None
        self._approach_count = 0
        self._near_mouth_count = 0
        self._inhalation_pattern_count = 0
        self._away_count = 0
        self._exhalation_pattern_count = 0
        self._exhalation_stability_count = 0

        self._mouth_opening_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._mouth_aspect_ratio_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._exhalation_opening_history = Smoother(window_size=self.EXHALATION_WINDOW)
        self._exhalation_width_history = Smoother(window_size=self.EXHALATION_WINDOW)
        self._exhalation_height_history = Smoother(window_size=self.EXHALATION_WINDOW)

        self._prev_mouth_opening = None
        self._prev_mouth_aspect_ratio = None
        self._prev_mouth_width = None
        self._prev_mouth_height = None
        self._inhalation_completed = False

        self.current_state = SmokingState.IDLE
        self.pattern_detected = False
        self.exhalation_detected = False

    def update(self, cigarette_tracker, cigarette_mouth_detector, face_tracker):
        cig_state = cigarette_mouth_detector.get_state()
        distance = cigarette_mouth_detector.get_distance()

        mouth_data = face_tracker.get_mouth_measurements()
        mouth_opening = mouth_data.get('opening', 0)
        mouth_aspect_ratio = mouth_data.get('aspect_ratio', 0)
        mouth_width = mouth_data.get('width', 0)
        mouth_height = mouth_data.get('height', 0)

        self._mouth_opening_history.add(mouth_opening)
        self._mouth_aspect_ratio_history.add(mouth_aspect_ratio)
        self._exhalation_opening_history.add(mouth_opening)
        self._exhalation_width_history.add(mouth_width)
        self._exhalation_height_history.add(mouth_height)

        self._update_state(cig_state, distance, mouth_opening, mouth_aspect_ratio, mouth_width, mouth_height)

        return self.current_state

    def _update_state(self, cig_state, distance, mouth_opening, mouth_aspect_ratio, mouth_width, mouth_height):
        if self._state == SmokingState.IDLE:
            self._frames_in_state = 0
            self._approach_count = 0
            self._near_mouth_count = 0
            self._inhalation_pattern_count = 0
            self._away_count = 0
            self._exhalation_pattern_count = 0
            self._exhalation_stability_count = 0
            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio
            self._prev_mouth_width = mouth_width
            self._prev_mouth_height = mouth_height
            self._inhalation_completed = False
            self.pattern_detected = False
            self.exhalation_detected = False

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
                    self._prev_mouth_width = mouth_width
                    self._prev_mouth_height = mouth_height
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
            self._prev_mouth_width = mouth_width
            self._prev_mouth_height = mouth_height

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
                    self._inhalation_completed = True
                    self.pattern_detected = True
                    self._inhalation_pattern_count = 0
            else:
                self._state = SmokingState.NEAR_MOUTH
                self._frames_in_state = 0
                self._inhalation_pattern_count = 0

            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio
            self._prev_mouth_width = mouth_width
            self._prev_mouth_height = mouth_height

        elif self._state == SmokingState.INHALING:
            self._frames_in_state += 1

            if cig_state == "MOVING_AWAY":
                self._away_count += 1
                if self._away_count >= self.AWAY_FRAME_COUNT:
                    self._state = SmokingState.EXHALATION_CANDIDATE
                    self._frames_in_state = 0
                    self._away_count = 0
                    self._exhalation_pattern_count = 0
                    self._exhalation_stability_count = 0
                    self._prev_mouth_opening = mouth_opening
                    self._prev_mouth_aspect_ratio = mouth_aspect_ratio
                    self._prev_mouth_width = mouth_width
                    self._prev_mouth_height = mouth_height
            elif cig_state == "FAR":
                self._state = SmokingState.EXHALATION_CANDIDATE
                self._frames_in_state = 0
                self._away_count = 0
                self._exhalation_pattern_count = 0
                self._exhalation_stability_count = 0
                self._prev_mouth_opening = mouth_opening
                self._prev_mouth_aspect_ratio = mouth_aspect_ratio
                self._prev_mouth_width = mouth_width
                self._prev_mouth_height = mouth_height
            else:
                self._away_count = 0

        elif self._state == SmokingState.EXHALATION_CANDIDATE:
            self._frames_in_state += 1

            if distance is None or distance > self.AWAY_FROM_MOUTH_THRESHOLD * 1.5:
                self._state = SmokingState.COMPLETED
                return

            if not face_tracker_is_valid(mouth_opening, mouth_width, mouth_height):
                self._exhalation_pattern_count = 0
                self._exhalation_stability_count = 0
            else:
                opening_delta = abs(mouth_opening - self._prev_mouth_opening) if self._prev_mouth_opening is not None else 0
                width_delta = abs(mouth_width - self._prev_mouth_width) if self._prev_mouth_width is not None else 0

                exhalation_pattern = (
                    mouth_opening > self.EXHALATION_MOUTH_OPENING_THRESHOLD and
                    (opening_delta > self.EXHALATION_MOUTH_OPENING_THRESHOLD * 0.5 or
                     width_delta > self.EXHALATION_MOUTH_WIDTH_CHANGE_THRESHOLD)
                )

                if exhalation_pattern:
                    self._exhalation_pattern_count += 1
                    self._exhalation_stability_count = 0
                    if self._exhalation_pattern_count >= self.EXHALATION_FRAME_COUNT:
                        self._state = SmokingState.EXHALING
                        self._frames_in_state = 0
                        self.exhalation_detected = True
                        self._exhalation_pattern_count = 0
                else:
                    self._exhalation_stability_count += 1
                    if self._exhalation_stability_count >= self.EXHALATION_STABILITY_FRAMES:
                        self._exhalation_pattern_count = max(0, self._exhalation_pattern_count - 1)

            self._prev_mouth_opening = mouth_opening
            self._prev_mouth_aspect_ratio = mouth_aspect_ratio
            self._prev_mouth_width = mouth_width
            self._prev_mouth_height = mouth_height

        elif self._state == SmokingState.EXHALING:
            self._frames_in_state += 1

            if distance is not None and distance < self.NEAR_MOUTH_THRESHOLD:
                self._state = SmokingState.NEAR_MOUTH
                self._frames_in_state = 0
                self.exhalation_detected = False
            elif distance is not None and distance > self.AWAY_FROM_MOUTH_THRESHOLD * 2:
                self._state = SmokingState.COMPLETED
                self._frames_in_state = 0

        elif self._state == SmokingState.COMPLETED:
            self._frames_in_state += 1
            if self._frames_in_state >= 2:
                self._state = SmokingState.IDLE

        self.current_state = self._state

    def get_state(self):
        return self.current_state

    def is_pattern_detected(self):
        return self.pattern_detected

    def is_exhalation_detected(self):
        return self.exhalation_detected

    def get_debug_info(self):
        return {
            'state': self.current_state,
            'pattern_detected': self.pattern_detected,
            'exhalation_detected': self.exhalation_detected,
            'frames_in_state': self._frames_in_state,
            'inhalation_completed': self._inhalation_completed,
            'thresholds': {
                'near_mouth': self.NEAR_MOUTH_THRESHOLD,
                'mouth_opening_change': self.MOUTH_OPENING_CHANGE_THRESHOLD,
                'mouth_ar_change': self.MOUTH_ASPECT_RATIO_CHANGE_THRESHOLD,
                'exhalation_opening': self.EXHALATION_MOUTH_OPENING_THRESHOLD,
                'exhalation_width_change': self.EXHALATION_MOUTH_WIDTH_CHANGE_THRESHOLD,
                'away_threshold': self.AWAY_FROM_MOUTH_THRESHOLD,
            },
            'frame_counts': {
                'approaching': self.APPROACHING_FRAME_COUNT,
                'near_mouth': self.NEAR_MOUTH_FRAME_COUNT,
                'inhalation': self.INHALATION_FRAME_COUNT,
                'away': self.AWAY_FRAME_COUNT,
                'exhalation': self.EXHALATION_FRAME_COUNT,
                'exhalation_stability': self.EXHALATION_STABILITY_FRAMES,
            }
        }

    def reset(self):
        self._state = SmokingState.IDLE
        self._frames_in_state = 0
        self._approach_count = 0
        self._near_mouth_count = 0
        self._inhalation_pattern_count = 0
        self._away_count = 0
        self._exhalation_pattern_count = 0
        self._exhalation_stability_count = 0
        self._mouth_opening_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._mouth_aspect_ratio_history = Smoother(window_size=self.INHALATION_WINDOW)
        self._exhalation_opening_history = Smoother(window_size=self.EXHALATION_WINDOW)
        self._exhalation_width_history = Smoother(window_size=self.EXHALATION_WINDOW)
        self._exhalation_height_history = Smoother(window_size=self.EXHALATION_WINDOW)
        self._prev_mouth_opening = None
        self._prev_mouth_aspect_ratio = None
        self._prev_mouth_width = None
        self._prev_mouth_height = None
        self._inhalation_completed = False
        self.current_state = SmokingState.IDLE
        self.pattern_detected = False
        self.exhalation_detected = False


def face_tracker_is_valid(opening, width, height):
    return opening > 0 and width > 0 and height > 0