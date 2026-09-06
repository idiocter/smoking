import cv2
import numpy as np
import os


class GlowEffect:
    def __init__(self, asset_path=None, max_intensity=1.0, fade_in_speed=0.15, fade_out_speed=0.08):
        self.max_intensity = max_intensity
        self.fade_in_speed = fade_in_speed
        self.fade_out_speed = fade_out_speed
        self.current_intensity = 0.0
        self.target_intensity = 0.0
        self.glow_img = None

        if asset_path is None:
            asset_path = os.path.join(os.path.dirname(__file__), '..', '..', 'assets', 'cigarette', 'cigarette_glow.png')

        self.load_asset(asset_path)

    def load_asset(self, asset_path):
        if os.path.exists(asset_path):
            self.glow_img = cv2.imread(asset_path, cv2.IMREAD_UNCHANGED)
            if self.glow_img is not None and self.glow_img.shape[2] == 3:
                self.glow_img = cv2.cvtColor(self.glow_img, cv2.COLOR_BGR2BGRA)
            print(f"Loaded glow asset: {asset_path}")
        else:
            print(f"Warning: Glow asset not found at {asset_path}")

    def set_target(self, should_glow):
        self.target_intensity = self.max_intensity if should_glow else 0.0

    def update(self):
        if self.current_intensity < self.target_intensity:
            self.current_intensity = min(self.target_intensity, self.current_intensity + self.fade_in_speed)
        elif self.current_intensity > self.target_intensity:
            self.current_intensity = max(self.target_intensity, self.current_intensity - self.fade_out_speed)
        return self.current_intensity

    def get_intensity(self):
        return self.current_intensity

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

        rotated = cv2.warpAffine(image, rot_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(0, 0, 0, 0))
        return rotated

    def _alpha_blend(self, frame, overlay, position):
        if overlay is None or overlay.size == 0 or self.current_intensity <= 0:
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

        roi = frame[y1:y2, x1:x2].astype(np.float32)
        overlay_roi = overlay[ov_y1:ov_y2, ov_x1:ov_x2].astype(np.float32)

        if overlay_roi.shape[2] == 4:
            alpha = (overlay_roi[:, :, 3:4] / 255.0) * self.current_intensity
            alpha = np.clip(alpha, 0, 1)
            blended = roi * (1 - alpha) + overlay_roi[:, :, :3] * alpha
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)

    def draw(self, frame, position, rotation, cigarette_length):
        if self.glow_img is None or self.current_intensity <= 0:
            return

        rotated = self._rotate_image(self.glow_img, rotation)

        # Position the glow at the cigarette tip (ember end)
        # The glow asset center is at the tip, so we offset by half cigarette length
        cos_r = np.cos(rotation)
        sin_r = np.sin(rotation)
        glow_center_x = int(position[0] + cos_r * (cigarette_length / 2))
        glow_center_y = int(position[1] + sin_r * (cigarette_length / 2))

        self._alpha_blend(frame, rotated, (glow_center_x, glow_center_y))