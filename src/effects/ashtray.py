import cv2


class AshtrayRenderer:
    """Screen-space ashtray with a catch area for the cigarette physics."""

    def __init__(self, width=210, height=72, margin=24):
        self.width = width
        self.height = height
        self.margin = margin
        self.center = None

    def update_layout(self, frame_width, frame_height):
        self.center = (
            frame_width - self.margin - self.width // 2,
            self.margin + self.height // 2,
        )

    def get_rest_pose(self):
        if self.center is None:
            return None
        return {
            'position': (float(self.center[0]), float(self.center[1] - 6)),
            'rotation': -0.12,
            'depth_rotation': 0.10,
        }

    def catches(self, position, previous_position=None):
        if self.center is None or position is None:
            return False

        half_width = self.width * 0.43
        rim_y = self.center[1] - 8
        inside_x = abs(position[0] - self.center[0]) <= half_width
        if not inside_x:
            return False

        if previous_position is None:
            return abs(position[1] - rim_y) <= self.height * 0.35
        crossed_rim = previous_position[1] <= rim_y <= position[1]
        already_inside = abs(position[1] - rim_y) <= self.height * 0.25
        return crossed_rim or already_inside

    def draw_back(self, frame):
        if self.center is None:
            return
        axes = (self.width // 2, self.height // 2)
        inner_axes = (int(self.width * 0.40), int(self.height * 0.28))
        cv2.ellipse(frame, self.center, axes, 0, 0, 360, (74, 77, 82), -1, cv2.LINE_AA)
        cv2.ellipse(frame, self.center, axes, 0, 0, 360, (155, 160, 168), 4, cv2.LINE_AA)
        cv2.ellipse(frame, (self.center[0], self.center[1] - 5), inner_axes,
                    0, 0, 360, (35, 37, 41), -1, cv2.LINE_AA)

    def draw_front(self, frame):
        if self.center is None:
            return
        axes = (self.width // 2, self.height // 2)
        cv2.ellipse(frame, self.center, axes, 0, 0, 180,
                    (190, 194, 201), 7, cv2.LINE_AA)
