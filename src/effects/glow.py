import numpy as np
import cv2


class GlowEffect:
    def __init__(self):
        self.intensity = 0.0
        self.target_intensity = 0.0
        self.rise_speed = 0.15
        self.fall_speed = 0.08

    def update(self, should_glow):
        self.target_intensity = 1.0 if should_glow else 0.0
        if self.intensity < self.target_intensity:
            self.intensity = min(self.target_intensity, self.intensity + self.rise_speed)
        else:
            self.intensity = max(self.target_intensity, self.intensity - self.fall_speed)
        return self.intensity

    def get_intensity(self):
        return self.intensity

    def draw(self, frame, position, rotation, length):
        if self.intensity <= 0:
            return

        center_x, center_y = int(position[0]), int(position[1])
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        ember_x = int(center_x + cos_r * (length / 2))
        ember_y = int(center_y + sin_r * (length / 2))

        for i in range(3):
            alpha = self.intensity * (1.0 - i * 0.25)
            radius = int(8 + i * 6 + self.intensity * 10)
            color = (0, int(100 * alpha), int(255 * alpha))
            cv2.circle(frame, (ember_x, ember_y), radius, color, -1)

        for i in range(5):
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, 15 * self.intensity)
            px = int(ember_x + dist * np.cos(angle))
            py = int(ember_y + dist * np.sin(angle))
            cv2.circle(frame, (px, py), 1, (0, 200, 255), -1)