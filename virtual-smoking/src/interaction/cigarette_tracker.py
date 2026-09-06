import numpy as np
from utils.geometry import distance, midpoint, vector_angle, normalize
from utils.smoothing import Smoother, OneEuroFilter


class CigaretteTracker:
    def __init__(self, smoothing_window=3, use_one_euro=True):
        self.length = 140
        self.thickness = 12
        self.is_held = False

        self._raw_position = None
        self._raw_rotation = 0.0

        self.position_smoother = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.3) if use_one_euro else Smoother(window_size=smoothing_window)
        self.rotation_smoother = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.5) if use_one_euro else Smoother(window_size=smoothing_window)

        self.position = None
        self.rotation = 0.0

        self._last_valid_position = None
        self._last_valid_rotation = 0.0
        self._frames_lost = 0
        self._max_frames_lost = 10

    def _calculate_cigarette_geometry(self, hand_landmarks):
        thumb_tip = hand_landmarks.get('thumb_tip')
        thumb_ip = hand_landmarks.get('thumb_ip')
        index_tip = hand_landmarks.get('index_tip')
        index_mcp = hand_landmarks.get('index_mcp')
        middle_tip = hand_landmarks.get('middle_tip')
        middle_mcp = hand_landmarks.get('middle_mcp')

        if not (thumb_tip and index_tip):
            return None, None

        # Position: midpoint between thumb tip and index tip
        position = midpoint(thumb_tip, index_tip)

        # Orientation: vector from thumb to index (primary grip direction)
        dx = index_tip[0] - thumb_tip[0]
        dy = index_tip[1] - thumb_tip[1]

        # If fingers are too close, use index finger direction as fallback
        if distance(thumb_tip, index_tip) < 15:
            if index_mcp and index_tip:
                dx = index_tip[0] - index_mcp[0]
                dy = index_tip[1] - index_mcp[1]
            elif middle_tip and middle_mcp:
                dx = middle_tip[0] - middle_mcp[0]
                dy = middle_tip[1] - middle_mcp[1]
            else:
                return position, self._raw_rotation

        rotation = vector_angle((dx, dy))

        # Normalize rotation to [-pi, pi]
        while rotation > np.pi:
            rotation -= 2 * np.pi
        while rotation < -np.pi:
            rotation += 2 * np.pi

        return position, rotation

    def update(self, hand_landmarks):
        if hand_landmarks is None or not hand_landmarks:
            self._frames_lost += 1
            if self._frames_lost > self._max_frames_lost:
                self.is_held = False
                self.position = None
            return

        self._frames_lost = 0

        raw_pos, raw_rot = self._calculate_cigarette_geometry(hand_landmarks)

        if raw_pos is None:
            self.is_held = False
            return

        self._raw_position = raw_pos
        self._raw_rotation = raw_rot

        # Apply smoothing
        if hasattr(self.position_smoother, '__call__'):
            # OneEuroFilter
            self.position = self.position_smoother(raw_pos)
            self.rotation = self.rotation_smoother(raw_rot)
        else:
            # Simple moving average
            self.position = self.position_smoother.add(raw_pos)
            self.rotation = self.rotation_smoother.add(raw_rot)

        if self.position is not None:
            self._last_valid_position = self.position
            self._last_valid_rotation = self.rotation
            self.is_held = True

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

    def get_orientation_vector(self):
        if self.position is None:
            return None
        return (np.cos(self.rotation), np.sin(self.rotation))

    def get_debug_info(self):
        return {
            'is_held': self.is_held,
            'position': self.position,
            'rotation': self.rotation,
            'rotation_deg': np.degrees(self.rotation) if self.rotation else 0,
            'raw_position': self._raw_position,
            'raw_rotation': self._raw_rotation,
            'frames_lost': self._frames_lost,
        }

    def reset(self):
        self.is_held = False
        self.position = None
        self.rotation = 0.0
        self._raw_position = None
        self._raw_rotation = 0.0
        self._last_valid_position = None
        self._last_valid_rotation = 0.0
        self._frames_lost = 0
        self.position_smoother = OneEuroFilter(freq=30.0, mincutoff=1.5, beta=0.3)
        self.rotation_smoother = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.5)