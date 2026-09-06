import numpy as np
from utils.geometry import distance, midpoint, vector_angle, normalize
from utils.smoothing import OneEuroFilter, OneEuroFilter2D, AngleOneEuroFilter
from config import Config


class CigaretteTracker:
    def __init__(self):
        self.length = Config.CIGARETTE_TRACKER['length']
        self.thickness = Config.CIGARETTE_TRACKER['thickness']
        self.is_held = False

        self._raw_position = None
        self._raw_rotation = 0.0

        pos_cfg = Config.CIGARETTE_TRACKER['position_smoothing']
        rot_cfg = Config.CIGARETTE_TRACKER['rotation_smoothing']

        self.position_smoother = OneEuroFilter2D(
            freq=pos_cfg['freq'],
            mincutoff=pos_cfg['mincutoff'],
            beta=pos_cfg['beta']
        )
        self.rotation_smoother = AngleOneEuroFilter(
            freq=rot_cfg['freq'],
            mincutoff=rot_cfg['mincutoff'],
            beta=rot_cfg['beta']
        )

        self.position = None
        self.rotation = 0.0

        self._last_valid_position = None
        self._last_valid_rotation = 0.0
        self._frames_lost = 0
        self._max_frames_lost = Config.CIGARETTE_TRACKER['max_frames_lost']
        self._min_finger_distance = Config.CIGARETTE_TRACKER['min_finger_distance']

    def _calculate_cigarette_geometry(self, hand_landmarks):
        thumb_tip = hand_landmarks.get('thumb_tip')
        thumb_ip = hand_landmarks.get('thumb_ip')
        index_tip = hand_landmarks.get('index_tip')
        index_mcp = hand_landmarks.get('index_mcp')
        middle_tip = hand_landmarks.get('middle_tip')
        middle_mcp = hand_landmarks.get('middle_mcp')

        if not (thumb_tip and index_tip):
            return None, None

        position = midpoint(thumb_tip, index_tip)

        dx = index_tip[0] - thumb_tip[0]
        dy = index_tip[1] - thumb_tip[1]

        if distance(thumb_tip, index_tip) < self._min_finger_distance:
            if index_mcp and index_tip:
                dx = index_tip[0] - index_mcp[0]
                dy = index_tip[1] - index_mcp[1]
            elif middle_tip and middle_mcp:
                dx = middle_tip[0] - middle_mcp[0]
                dy = middle_tip[1] - middle_mcp[1]
            else:
                return position, self._raw_rotation

        rotation = vector_angle((dx, dy))

        return position, rotation

    def update(self, hand_landmarks):
        if hand_landmarks is None or not hand_landmarks:
            self._frames_lost += 1
            if self._frames_lost > self._max_frames_lost:
                self.is_held = False
                self.position = None
                self.position_smoother.reset()
                self.rotation_smoother.reset()
            return

        self._frames_lost = 0

        raw_pos, raw_rot = self._calculate_cigarette_geometry(hand_landmarks)

        if raw_pos is None:
            self.is_held = False
            return

        self._raw_position = raw_pos
        self._raw_rotation = raw_rot

        self.position = self.position_smoother(raw_pos)
        self.rotation = self.rotation_smoother(raw_rot)

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

    def get_mouth_end_position(self, mouth_center=None):
        if self.position is None:
            return None

        tip = self.get_tip_position()
        base = self.get_base_position()

        if mouth_center is None:
            return tip

        tip_dist = distance(tip, mouth_center)
        base_dist = distance(base, mouth_center)

        return tip if tip_dist < base_dist else base

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
        pos_cfg = Config.CIGARETTE_TRACKER['position_smoothing']
        rot_cfg = Config.CIGARETTE_TRACKER['rotation_smoothing']
        self.position_smoother = OneEuroFilter2D(
            freq=pos_cfg['freq'],
            mincutoff=pos_cfg['mincutoff'],
            beta=pos_cfg['beta']
        )
        self.rotation_smoother = AngleOneEuroFilter(
            freq=rot_cfg['freq'],
            mincutoff=rot_cfg['mincutoff'],
            beta=rot_cfg['beta']
        )