from enum import Enum


class SmokingState(Enum):
    IDLE = "IDLE"
    CIGARETTE_MOVING_TO_MOUTH = "CIGARETTE_MOVING_TO_MOUTH"
    CIGARETTE_NEAR_MOUTH = "CIGARETTE_NEAR_MOUTH"
    MOUTH_PATTERN_DETECTED = "MOUTH_PATTERN_DETECTED"
    INHALATION_PATTERN = "INHALATION_PATTERN"
    CIGARETTE_GLOW = "CIGARETTE_GLOW"
    CIGARETTE_MOVING_AWAY = "CIGARETTE_MOVING_AWAY"
    EXHALATION_PATTERN = "EXHALATION_PATTERN"
    SMOKE_EFFECT = "SMOKE_EFFECT"


class SmokingDetector:
    def __init__(self):
        self.state = SmokingState.IDLE
        self.frames_in_state = 0
        self.cigarette_near_mouth_threshold = 80
        self.mouth_open_threshold = 15
        self.movement_threshold = 5

    def update(self, cigarette_pos, mouth_pos, mouth_opening, prev_cigarette_pos=None):
        self.frames_in_state += 1

        if self.state == SmokingState.IDLE:
            if cigarette_pos and mouth_pos:
                dist = ((cigarette_pos[0] - mouth_pos[0])**2 + (cigarette_pos[1] - mouth_pos[1])**2)**0.5
                if dist < self.cigarette_near_mouth_threshold * 2:
                    self.state = SmokingState.CIGARETTE_MOVING_TO_MOUTH
                    self.frames_in_state = 0

        elif self.state == SmokingState.CIGARETTE_MOVING_TO_MOUTH:
            if cigarette_pos and mouth_pos:
                dist = ((cigarette_pos[0] - mouth_pos[0])**2 + (cigarette_pos[1] - mouth_pos[1])**2)**0.5
                if dist < self.cigarette_near_mouth_threshold:
                    self.state = SmokingState.CIGARETTE_NEAR_MOUTH
                    self.frames_in_state = 0
                elif dist > self.cigarette_near_mouth_threshold * 3:
                    self.state = SmokingState.IDLE
                    self.frames_in_state = 0

        elif self.state == SmokingState.CIGARETTE_NEAR_MOUTH:
            if mouth_opening > self.mouth_open_threshold:
                self.state = SmokingState.MOUTH_PATTERN_DETECTED
                self.frames_in_state = 0

        elif self.state == SmokingState.MOUTH_PATTERN_DETECTED:
            if self.frames_in_state > 3:
                self.state = SmokingState.INHALATION_PATTERN
                self.frames_in_state = 0

        elif self.state == SmokingState.INHALATION_PATTERN:
            if self.frames_in_state > 5:
                self.state = SmokingState.CIGARETTE_GLOW
                self.frames_in_state = 0

        elif self.state == SmokingState.CIGARETTE_GLOW:
            if cigarette_pos and mouth_pos:
                dist = ((cigarette_pos[0] - mouth_pos[0])**2 + (cigarette_pos[1] - mouth_pos[1])**2)**0.5
                if dist > self.cigarette_near_mouth_threshold * 1.5:
                    self.state = SmokingState.CIGARETTE_MOVING_AWAY
                    self.frames_in_state = 0

        elif self.state == SmokingState.CIGARETTE_MOVING_AWAY:
            if mouth_opening > self.mouth_open_threshold:
                self.state = SmokingState.EXHALATION_PATTERN
                self.frames_in_state = 0

        elif self.state == SmokingState.EXHALATION_PATTERN:
            if self.frames_in_state > 3:
                self.state = SmokingState.SMOKE_EFFECT
                self.frames_in_state = 0

        elif self.state == SmokingState.SMOKE_EFFECT:
            if self.frames_in_state > 10:
                self.state = SmokingState.IDLE
                self.frames_in_state = 0

        return self.state

    def get_state(self):
        return self.state

    def reset(self):
        self.state = SmokingState.IDLE
        self.frames_in_state = 0