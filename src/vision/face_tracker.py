import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from utils.smoothing import Smoother
from config import Config


class FaceTracker:
    def __init__(self, max_faces=1, min_detection_confidence=0.5,
                 min_tracking_confidence=0.5, model_asset_path=None):
        if model_asset_path is None:
            model_asset_path = Path(__file__).resolve().parents[2] / 'face_landmarker.task'
        base_options = python.BaseOptions(
            model_asset_path=str(model_asset_path),
            delegate=python.BaseOptions.Delegate.CPU
        )
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_faces=max_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(options)

        self.landmark_indices = {
            'nose': 1,
            'upper_lip': 13,
            'lower_lip': 14,
            'mouth_left': 61,
            'mouth_right': 291,
            'mouth_center': 0,
            'chin': 152,
            'left_eye': 33,
            'right_eye': 263,
            'left_eyebrow': 70,
            'right_eyebrow': 300,
            'upper_lip_top': 12,
            'lower_lip_bottom': 15,
        }

        self._landmarks = None
        self._image_shape = None
        self._timestamp = 0

        # Smoothing for mouth measurements
        window = Config.FACE_TRACKER['mouth_smoothing_window']
        self._mouth_center_smoother = Smoother(window_size=window)
        self._mouth_width_smoother = Smoother(window_size=window)
        self._mouth_height_smoother = Smoother(window_size=window)
        self._mouth_opening_smoother = Smoother(window_size=window)
        self._mouth_aspect_ratio_smoother = Smoother(window_size=window)

    def process(self, frame):
        self._image_shape = frame.shape[:2]
        scale = Config.FACE_TRACKER.get('processing_scale', 1.0)
        tracking_frame = frame
        if 0 < scale < 1.0:
            tracking_frame = cv2.resize(
                frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA
            )
        rgb = cv2.cvtColor(tracking_frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self._timestamp += 1
        result = self.landmarker.detect_for_video(mp_image, self._timestamp)
        self._landmarks = None
        if result.face_landmarks:
            self._landmarks = result.face_landmarks[0]
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

    def get_mouth_center(self):
        left = self.get_landmark('mouth_left')
        right = self.get_landmark('mouth_right')
        upper = self.get_landmark('upper_lip')
        lower = self.get_landmark('lower_lip')
        if left and right and upper and lower:
            cx = (left[0] + right[0]) / 2
            cy = (upper[1] + lower[1]) / 2
            return self._mouth_center_smoother.add((cx, cy))
        if left and right:
            cx = (left[0] + right[0]) / 2
            cy = (left[1] + right[1]) / 2
            return self._mouth_center_smoother.add((cx, cy))
        self._mouth_center_smoother.clear()
        return None

    def get_mouth_opening(self):
        upper = self.get_landmark('upper_lip')
        lower = self.get_landmark('lower_lip')
        if upper and lower:
            opening = abs(lower[1] - upper[1])
            return self._mouth_opening_smoother.add(opening)
        self._mouth_opening_smoother.clear()
        return 0

    def get_mouth_width(self):
        left = self.get_landmark('mouth_left')
        right = self.get_landmark('mouth_right')
        if left and right:
            width = abs(right[0] - left[0])
            return self._mouth_width_smoother.add(width)
        self._mouth_width_smoother.clear()
        return 0

    def get_mouth_height(self):
        upper = self.get_landmark('upper_lip_top')
        lower = self.get_landmark('lower_lip_bottom')
        if upper and lower:
            height = abs(lower[1] - upper[1])
            return self._mouth_height_smoother.add(height)
        self._mouth_height_smoother.clear()
        return self.get_mouth_opening()

    def get_mouth_aspect_ratio(self):
        width = self.get_mouth_width()
        height = self.get_mouth_height()
        if height > 0:
            ar = width / height
            return self._mouth_aspect_ratio_smoother.add(ar)
        self._mouth_aspect_ratio_smoother.clear()
        return 0.0

    def get_mouth_measurements(self):
        return {
            'center': self.get_mouth_center(),
            'opening': self.get_mouth_opening(),
            'width': self.get_mouth_width(),
            'height': self.get_mouth_height(),
            'aspect_ratio': self.get_mouth_aspect_ratio(),
        }

    def get_breath_direction(self):
        """Estimate the screen-space direction in which the face is pointing."""
        nose = self.get_landmark('nose')
        left_eye = self.get_landmark('left_eye')
        right_eye = self.get_landmark('right_eye')
        if not nose or not left_eye or not right_eye:
            return (0.0, -0.58)

        eye_mid_x = (left_eye[0] + right_eye[0]) * 0.5
        eye_distance = max(1.0, abs(right_eye[0] - left_eye[0]))
        yaw = float(np.clip(
            (nose[0] - eye_mid_x) / (eye_distance * 0.28), -1.0, 1.0
        ))
        # Frontal breath travels toward the camera and is represented by rapid
        # expansion. Horizontal travel appears only when the head turns.
        return (yaw, -0.58)

    def is_detected(self):
        return self._landmarks is not None

    def draw_landmarks(self, frame, draw_connections=True):
        if self._landmarks is None:
            return

        for name, idx in self.landmark_indices.items():
            pt = self.get_landmark(name)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                cv2.putText(frame, name, (int(pt[0]) + 5, int(pt[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    def close(self):
        self.landmarker.close()
