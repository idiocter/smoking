import cv2
import numpy as np


class CigaretteRenderer:
    def __init__(self, length=120, thickness=8):
        self.length = length
        self.thickness = thickness
        self.color = (40, 40, 40)
        self.filter_color = (180, 150, 100)

    def draw(self, frame, position, rotation, glow_intensity=0.0):
        if position is None:
            return

        center_x, center_y = int(position[0]), int(position[1])
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        half_len = self.length // 2
        half_thick = self.thickness // 2

        corners = np.array([
            [-half_len, -half_thick],
            [half_len, -half_thick],
            [half_len, half_thick],
            [-half_len, half_thick]
        ], dtype=np.float32)

        rot_matrix = np.array([[cos_r, -sin_r], [sin_r, cos_r]], dtype=np.float32)
        rotated = corners @ rot_matrix.T
        rotated[:, 0] += center_x
        rotated[:, 1] += center_y
        pts = rotated.astype(np.int32)

        cv2.fillPoly(frame, [pts], self.color)

        filter_len = self.length // 6
        filter_corners = np.array([
            [half_len - filter_len, -half_thick],
            [half_len, -half_thick],
            [half_len, half_thick],
            [half_len - filter_len, half_thick]
        ], dtype=np.float32)
        filter_rotated = filter_corners @ rot_matrix.T
        filter_rotated[:, 0] += center_x
        filter_rotated[:, 1] += center_y
        filter_pts = filter_rotated.astype(np.int32)
        cv2.fillPoly(frame, [filter_pts], self.filter_color)

        if glow_intensity > 0:
            ember_pos = (int(center_x + cos_r * half_len), int(center_y + sin_r * half_len))
            radius = int(6 + glow_intensity * 8)
            glow_color = (0, int(100 + glow_intensity * 155), int(255 * glow_intensity))
            cv2.circle(frame, ember_pos, radius, glow_color, -1)
            cv2.circle(frame, ember_pos, radius + 3, (0, 50, int(200 * glow_intensity)), 2)