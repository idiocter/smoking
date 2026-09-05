import numpy as np
from utils.geometry import distance, angle_between, midpoint, vector_angle, normalize


class CigaretteTracker:
    def __init__(self):
        self.position = None
        self.rotation = 0.0
        self.length = 120
        self.is_held = False

    def update(self, hand_landmarks):
        if hand_landmarks is None:
            self.is_held = False
            return

        thumb_tip = hand_landmarks.get('thumb_tip')
        index_tip = hand_landmarks.get('index_tip')
        middle_tip = hand_landmarks.get('middle_tip')

        if thumb_tip and index_tip:
            self.position = midpoint(thumb_tip, index_tip)
            dx = index_tip[0] - thumb_tip[0]
            dy = index_tip[1] - thumb_tip[1]
            self.rotation = vector_angle((dx, dy))
            self.is_held = True
        else:
            self.is_held = False

    def get_tip_position(self):
        if self.position is None:
            return None
        dx = np.cos(self.rotation) * (self.length / 2)
        dy = np.sin(self.rotation) * (self.length / 2)
        return (self.position[0] + dx, self.position[1] + dy)

    def get_base_position(self):
        if self.position is None:
            return None
        dx = -np.cos(self.rotation) * (self.length / 2)
        dy = -np.sin(self.rotation) * (self.length / 2)
        return (self.position[0] + dx, self.position[1] + dy)