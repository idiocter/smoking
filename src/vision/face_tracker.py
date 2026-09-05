import cv2
import mediapipe as mp
import numpy as np


class FaceTracker:
    def __init__(self, max_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles

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
        }

        self._landmarks = None
        self._image_shape = None

    def process(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._image_shape = frame.shape[:2]
        results = self.face_mesh.process(rgb)
        self._landmarks = None
        if results.multi_face_landmarks:
            self._landmarks = results.multi_face_landmarks[0]
        return self._landmarks is not None

    def get_landmark(self, name):
        if self._landmarks is None or name not in self.landmark_indices:
            return None
        idx = self.landmark_indices[name]
        lm = self._landmarks.landmark[idx]
        h, w = self._image_shape
        return (lm.x * w, lm.y * h)

    def get_landmarks(self, names):
        return {name: self.get_landmark(name) for name in names}

    def get_all_landmarks(self):
        if self._landmarks is None:
            return None
        h, w = self._image_shape
        return [(lm.x * w, lm.y * h) for lm in self._landmarks.landmark]

    def get_mouth_center(self):
        left = self.get_landmark('mouth_left')
        right = self.get_landmark('mouth_right')
        if left and right:
            return ((left[0] + right[0]) / 2, (left[1] + right[1]) / 2)
        return None

    def get_mouth_opening(self):
        upper = self.get_landmark('upper_lip')
        lower = self.get_landmark('lower_lip')
        if upper and lower:
            return abs(lower[1] - upper[1])
        return 0

    def is_detected(self):
        return self._landmarks is not None

    def draw_landmarks(self, frame, draw_connections=True):
        if self._landmarks is None:
            return
        if draw_connections:
            self.mp_drawing.draw_landmarks(
                frame,
                self._landmarks,
                self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            self.mp_drawing.draw_landmarks(
                frame,
                self._landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )

        for name, idx in self.landmark_indices.items():
            pt = self.get_landmark(name)
            if pt:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
                cv2.putText(frame, name, (int(pt[0]) + 5, int(pt[1]) - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

    def close(self):
        self.face_mesh.close()