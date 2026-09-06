import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class HandTracker:
    def __init__(self, max_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        base_options = python.BaseOptions(
            model_asset_path='hand_landmarker.task',
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

        self.landmark_indices = {
            'wrist': 0,
            'thumb_cmc': 1,
            'thumb_mcp': 2,
            'thumb_ip': 3,
            'thumb_tip': 4,
            'index_mcp': 5,
            'index_pip': 6,
            'index_dip': 7,
            'index_tip': 8,
            'middle_mcp': 9,
            'middle_pip': 10,
            'middle_dip': 11,
            'middle_tip': 12,
            'ring_mcp': 13,
            'ring_pip': 14,
            'ring_dip': 15,
            'ring_tip': 16,
            'pinky_mcp': 17,
            'pinky_pip': 18,
            'pinky_dip': 19,
            'pinky_tip': 20,
        }

        self._landmarks = None
        self._handedness = None
        self._image_shape = None
        self._timestamp = 0

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._image_shape = frame.shape[:2]
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 1
        result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        self._landmarks = None
        self._handedness = None
        if result.hand_landmarks:
            self._landmarks = result.hand_landmarks[0]
            if result.handedness:
                self._handedness = result.handedness[0][0].category_name
        return self._landmarks is not None

    def get_landmark(self, name):
        if self._landmarks is None or name not in self.landmark_indices:
            return None
        idx = self.landmark_indices[name]
        lm = self._landmarks[idx]
        h, w = self._image_shape
        return (lm.x * w, lm.y * h)

    def get_landmarks(self, names):
        return {name: self.get_landmark(name) for name in names}

    def get_all_landmarks(self):
        if self._landmarks is None:
            return None
        h, w = self._image_shape
        return [(lm.x * w, lm.y * h) for lm in self._landmarks]

    def get_handedness(self):
        return self._handedness

    def is_detected(self):
        return self._landmarks is not None

    def get_index_finger_direction(self):
        mcp = self.get_landmark('index_mcp')
        tip = self.get_landmark('index_tip')
        if mcp and tip:
            dx = tip[0] - mcp[0]
            dy = tip[1] - mcp[1]
            norm = (dx**2 + dy**2)**0.5
            if norm > 0:
                return (dx / norm, dy / norm)
        return None

    def get_thumb_index_midpoint(self):
        thumb_tip = self.get_landmark('thumb_tip')
        index_tip = self.get_landmark('index_tip')
        if thumb_tip and index_tip:
            return ((thumb_tip[0] + index_tip[0]) / 2, (thumb_tip[1] + index_tip[1]) / 2)
        return None

    def get_fingertip_positions(self):
        tips = ['thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        return {name: self.get_landmark(name) for name in tips}

    def draw_landmarks(self, frame, draw_connections=True):
        if self._landmarks is None:
            return

        for name, idx in self.landmark_indices.items():
            pt = self.get_landmark(name)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (255, 0, 0), -1)
                cv2.putText(frame, name, (int(pt[0]) + 5, int(pt[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 0, 0), 1)

    def close(self):
        self.landmarker.close()