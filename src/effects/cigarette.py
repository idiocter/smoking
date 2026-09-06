import cv2
import numpy as np
import os


class CigaretteRenderer:
    def __init__(self, asset_path=None, length=140, thickness=12):
        self.length = length
        self.thickness = thickness
        self.cigarette_img = None

        if asset_path is None:
            asset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'cigarette', 'cigarette.png')

        self.load_asset(asset_path)

    def load_asset(self, asset_path):
        if os.path.exists(asset_path):
            self.cigarette_img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
            if self.cigarette_img is not None:
                h, w = self.cigarette_img.shape[:2]
                if self.cigarette_img.shape[2] == 3:
                    self.cigarette_img = cv2.cvtColor(self.cigarette_img, cv2.COLOR_BGR2BGRA)
                self._scale_asset()
                print(f"Loaded cigarette asset: {asset_path} ({w}x{h})")
            else:
                print(f"Warning: Could not load cigarette asset from {asset_path}")
        else:
            print(f"Warning: Cigarette asset not found at {asset_path}")

    def _scale_asset(self):
        if self.cigarette_img is None:
            return
        h, w = self.cigarette_img.shape[:2]
        scale = self.length / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        self.cigarette_img = cv2.resize(self.cigarette_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    def _rotate_image(self, image, angle):
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        rot_matrix = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)

        cos = np.abs(rot_matrix[0, 0])
        sin = np.abs(rot_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        rot_matrix[0, 2] += (new_w / 2) - center[0]
        rot_matrix[1, 2] += (new_h / 2) - center[1]

        rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
        return rotated

    def _alpha_blend(self, frame, overlay, position):
        if overlay is None or overlay.size == 0:
            return

        h, w = overlay.shape[:2]
        x, y = int(position[0] - w // 2), int(position[1] - h // 2)

        frame_h, frame_w = frame.shape[:2]

        if x >= frame_w or y >= frame_h or x + w <= 0 or y + h <= 0:
            return

        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + w), min(frame_h, y + h)
        ov_x1, ov_y1 = x1 - x, y1 - y
        ov_x2, ov_y2 = ov_x1 + (x2 - x1), ov_y1 + (y2 - y1)

        if ov_x2 <= ov_x1 or ov_y2 <= ov_y1:
            return

        roi = frame[y1:y2, x1:x2]
        overlay_roi = overlay[ov_y1:ov_y2, ov_x1:ov_x2]

        if overlay_roi.shape[2] == 4:
            alpha = overlay_roi[:, :, 3:4] / 255.0
            alpha = np.clip(alpha, 0, 1)
            frame[y1:y2, x1:x2] = (roi * (1 - alpha) + overlay_roi[:, :, :3] * alpha).astype(np.uint8)

    def draw(self, frame, position, rotation, glow_intensity=0.0):
        if position is None or self.cigarette_img is None:
            return

        rotated = self._rotate_image(self.cigarette_img, rotation)
        self._alpha_blend(frame, rotated, position)

        if glow_intensity > 0:
            self._draw_glow(frame, position, rotation, glow_intensity)

    def _draw_glow(self, frame, position, rotation, intensity):
        center_x, center_y = int(position[0]), int(position[1])
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)

        ember_x = int(center_x + cos_r * (self.length / 2))
        ember_y = int(center_y + sin_r * (self.length / 2))

        radius = int(6 + intensity * 8)
        glow_color = (0, int(100 + intensity * 155), int(255 * intensity))
        cv2.circle(frame, (ember_x, ember_y), radius, glow_color, -1)
        cv2.circle(frame, (ember_x, ember_y), radius + 3, (0, 50, int(200 * intensity)), 2)


class CigaretteRendererFallback:
    def __init__(self, length=140, thickness=12):
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