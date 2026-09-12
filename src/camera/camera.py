import cv2
import time


class Camera:
    def __init__(self, device_index=0, width=1280, height=720, fps=30):
        self.device_index = device_index
        self.width = width
        self.height = height
        self.target_fps = fps
        self.cap = None
        self.fps = 0
        self._prev_time = 0
        self._frame_count = 0

    def open(self):
        self.cap = cv2.VideoCapture(self.device_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera device {self.device_index}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._prev_time = time.time()
        return True

    def read(self):
        if self.cap is None or not self.cap.isOpened():
            return None
        ret, frame = self.cap.read()
        if not ret:
            return None
        frame = cv2.flip(frame, 1)
        self._update_fps()
        return frame

    def _update_fps(self):
        self._frame_count += 1
        now = time.time()
        elapsed = now - self._prev_time
        if elapsed >= 1.0:
            self.fps = self._frame_count / elapsed
            self._frame_count = 0
            self._prev_time = now

    def get_fps(self):
        return self.fps

    def get_frame_size(self):
        if self.cap is None:
            return (self.width, self.height)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
