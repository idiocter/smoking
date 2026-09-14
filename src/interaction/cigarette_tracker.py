import numpy as np
from utils.geometry import distance, midpoint, vector_angle
from utils.smoothing import OneEuroFilter, OneEuroFilter2D, AngleOneEuroFilter
from config import Config


class CigaretteTracker:
    RESTING = "RESTING"
    HELD = "HELD"
    FALLING = "FALLING"

    def __init__(self):
        self.base_length = Config.CIGARETTE_TRACKER['length']
        self.length = self.base_length
        self.thickness = Config.CIGARETTE_TRACKER['thickness']
        self.is_held = False

        self._raw_position = None
        self._raw_rotation = 0.0
        self._raw_depth_rotation = Config.CIGARETTE_TRACKER['default_depth_rotation']

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
        self.depth_smoother = AngleOneEuroFilter(
            freq=rot_cfg['freq'],
            mincutoff=rot_cfg['mincutoff'],
            beta=rot_cfg['beta']
        )
        self.length_smoother = OneEuroFilter(
            freq=pos_cfg['freq'],
            mincutoff=pos_cfg['mincutoff'],
            beta=pos_cfg['beta']
        )

        self.position = None
        self.rotation = 0.0
        self.depth_rotation = Config.CIGARETTE_TRACKER['default_depth_rotation']

        self._last_valid_position = None
        self._last_valid_rotation = 0.0
        self._frames_lost = 0
        self._max_frames_lost = Config.CIGARETTE_TRACKER['max_frames_lost']
        self._min_finger_distance = Config.CIGARETTE_TRACKER['min_finger_distance']
        self._grip_tip_ratio = Config.CIGARETTE_TRACKER['grip_tip_ratio']
        self.state = self.RESTING
        self.velocity = np.zeros(2, dtype=np.float64)
        self.angular_velocity = 0.0
        self._physics_initialized = False
        self._grab_position_offset = np.zeros(2, dtype=np.float64)
        self._grab_rotation_offset = 0.0
        self._grab_depth_offset = 0.0

    @staticmethod
    def _interpolate(start, end, amount):
        return tuple(start[i] + (end[i] - start[i]) * amount for i in range(len(start)))

    def _grip_point(self, landmarks, finger):
        tip = landmarks.get(f'{finger}_tip')
        dip = landmarks.get(f'{finger}_dip')
        if tip is None:
            return None
        if dip is None:
            return tip
        return self._interpolate(dip, tip, self._grip_tip_ratio)

    def _calculate_depth_rotation(self, hand_landmarks_3d, fallback_rotation):
        if not hand_landmarks_3d:
            return fallback_rotation, Config.CIGARETTE_TRACKER['default_depth_rotation']

        wrist = hand_landmarks_3d.get('wrist')
        index_mcp = hand_landmarks_3d.get('index_mcp')
        middle_mcp = hand_landmarks_3d.get('middle_mcp')
        pinky_mcp = hand_landmarks_3d.get('pinky_mcp')
        if not all((wrist, index_mcp, middle_mcp, pinky_mcp)):
            return fallback_rotation, Config.CIGARETTE_TRACKER['default_depth_rotation']

        across_palm = np.subtract(pinky_mcp, index_mcp)
        along_palm = np.subtract(middle_mcp, wrist)
        palm_normal = np.cross(across_palm, along_palm)
        normal_length = np.linalg.norm(palm_normal)
        if normal_length < 1e-6:
            return fallback_rotation, Config.CIGARETTE_TRACKER['default_depth_rotation']

        palm_normal = palm_normal / normal_length
        # MediaPipe uses negative Z toward the camera. Keep the cigarette's
        # outward end on that side regardless of left/right handedness.
        if palm_normal[2] > 0:
            palm_normal = -palm_normal

        screen_length = np.hypot(palm_normal[0], palm_normal[1])
        rotation = fallback_rotation
        if screen_length > 0.08:
            rotation = vector_angle((palm_normal[0], palm_normal[1]))

        depth_rotation = np.arctan2(-palm_normal[2], screen_length)
        depth_rotation = np.clip(
            depth_rotation,
            0.0,
            Config.CIGARETTE_TRACKER['max_depth_rotation'],
        )
        return rotation, depth_rotation

    def _calculate_hand_length(self, hand_landmarks):
        wrist = hand_landmarks.get('wrist')
        middle_tip = hand_landmarks.get('middle_tip')
        if wrist is None or middle_tip is None:
            return Config.CIGARETTE_TRACKER['reference_hand_length']
        return distance(wrist, middle_tip)

    def _calculate_length(self, hand_landmarks):
        hand_length = self._calculate_hand_length(hand_landmarks)
        scaled_length = self.base_length * (
            hand_length / Config.CIGARETTE_TRACKER['reference_hand_length']
        )
        return np.clip(
            scaled_length,
            Config.CIGARETTE_TRACKER['min_length'],
            Config.CIGARETTE_TRACKER['max_length'],
        )

    def _calculate_cigarette_geometry(self, hand_landmarks, hand_landmarks_3d=None):
        index_tip = self._grip_point(hand_landmarks, 'index')
        index_mcp = hand_landmarks.get('index_mcp')
        middle_tip = self._grip_point(hand_landmarks, 'middle')
        middle_mcp = hand_landmarks.get('middle_mcp')

        if not (index_tip and middle_tip):
            return None, None, None, None, None

        hand_length = self._calculate_hand_length(hand_landmarks)
        grip_distance = distance(index_tip, middle_tip)
        grip_ratio = grip_distance / max(hand_length, 1.0)

        position = midpoint(index_tip, middle_tip)

        dx = middle_tip[0] - index_tip[0]
        dy = middle_tip[1] - index_tip[1]

        if grip_distance < self._min_finger_distance:
            finger_directions = []
            if index_mcp:
                finger_directions.append((
                    index_tip[0] - index_mcp[0],
                    index_tip[1] - index_mcp[1],
                ))
            if middle_mcp:
                finger_directions.append((
                    middle_tip[0] - middle_mcp[0],
                    middle_tip[1] - middle_mcp[1],
                ))

            if not finger_directions:
                return (
                    position,
                    self._raw_rotation,
                    self._raw_depth_rotation,
                    self.base_length,
                    grip_ratio,
                )

            avg_dx = sum(direction[0] for direction in finger_directions) / len(finger_directions)
            avg_dy = sum(direction[1] for direction in finger_directions) / len(finger_directions)
            dx, dy = avg_dy, -avg_dx

            if dx == 0 and dy == 0:
                return (
                    position,
                    self._raw_rotation,
                    self._raw_depth_rotation,
                    self.base_length,
                    grip_ratio,
                )

        rotation = vector_angle((dx, dy))
        rotation, depth_rotation = self._calculate_depth_rotation(
            hand_landmarks_3d, rotation
        )
        length = self._calculate_length(hand_landmarks)

        return position, rotation, depth_rotation, length, grip_ratio

    def _apply_tracked_pose(self, raw_pos, raw_rot, raw_depth, raw_length):
        self._raw_position = raw_pos
        self._raw_rotation = raw_rot
        self._raw_depth_rotation = raw_depth
        self.position = self.position_smoother(raw_pos)
        self.rotation = self.rotation_smoother(raw_rot)
        self.depth_rotation = self.depth_smoother(raw_depth)
        self.length = self.length_smoother(raw_length)
        self._last_valid_position = self.position
        self._last_valid_rotation = self.rotation
        self.is_held = True
        self.state = self.HELD

    def _place_in_ashtray(self, ashtray):
        pose = ashtray.get_rest_pose() if ashtray is not None else None
        if pose is None:
            return
        self.position = pose['position']
        self.rotation = pose['rotation']
        self.depth_rotation = pose['depth_rotation']
        self.velocity[:] = 0.0
        self.angular_velocity = 0.0
        self.is_held = False
        self.state = self.RESTING
        self._physics_initialized = True

    def _begin_hold(self, raw_pos, raw_rot, raw_depth):
        self._grab_position_offset = np.subtract(self.position, raw_pos)
        self._grab_rotation_offset = self.rotation - raw_rot
        self._grab_depth_offset = self.depth_rotation - raw_depth
        self.velocity[:] = 0.0
        self.angular_velocity = 0.0
        self.is_held = True
        self.state = self.HELD
        self._frames_lost = 0

    def _release(self):
        self.is_held = False
        self.state = self.FALLING

    def _update_falling(self, frame_shape, ashtray, dt):
        if self.position is None:
            self._place_in_ashtray(ashtray)
            return
        previous_position = self.position
        cfg = Config.CIGARETTE_TRACKER
        self.velocity[1] += cfg['gravity'] * dt
        self.velocity *= cfg['air_drag'] ** dt
        self.position = tuple(np.add(self.position, self.velocity * dt))
        self.rotation += self.angular_velocity * dt
        self.angular_velocity *= cfg['angular_drag'] ** dt

        if ashtray is not None and ashtray.catches(self.position, previous_position):
            self._place_in_ashtray(ashtray)
            return

        if frame_shape is not None:
            frame_height = frame_shape[0]
            if self.position[1] > frame_height + cfg['respawn_margin']:
                self._place_in_ashtray(ashtray)

    def update(self, hand_landmarks, hand_landmarks_3d=None,
               frame_shape=None, ashtray=None, dt=1.0):
        physics_enabled = frame_shape is not None and ashtray is not None
        if physics_enabled and not self._physics_initialized:
            self._place_in_ashtray(ashtray)

        if hand_landmarks is None or not hand_landmarks:
            self._frames_lost += 1
            if physics_enabled:
                if self.is_held and self._frames_lost > self._max_frames_lost:
                    self._release()
                if self.state == self.FALLING:
                    self._update_falling(frame_shape, ashtray, dt)
            elif self._frames_lost > self._max_frames_lost:
                self.is_held = False
                self.position = None
                self.position_smoother.reset()
                self.rotation_smoother.reset()
                self.depth_smoother.reset()
                self.length_smoother.reset()
            return

        self._frames_lost = 0

        raw_pos, raw_rot, raw_depth, raw_length, grip_ratio = (
            self._calculate_cigarette_geometry(hand_landmarks, hand_landmarks_3d)
        )

        if raw_pos is None:
            if physics_enabled and self.state == self.FALLING:
                self._update_falling(frame_shape, ashtray, dt)
            elif not physics_enabled:
                self.is_held = False
            return

        if not physics_enabled:
            if grip_ratio <= Config.CIGARETTE_TRACKER['max_grip_distance_ratio']:
                self._apply_tracked_pose(raw_pos, raw_rot, raw_depth, raw_length)
            else:
                self.is_held = False
            return

        cfg = Config.CIGARETTE_TRACKER
        if self.state == self.HELD:
            if grip_ratio > cfg['release_grip_distance_ratio']:
                self._release()
                self._update_falling(frame_shape, ashtray, dt)
                return

            target_position = np.add(raw_pos, self._grab_position_offset)
            displacement = target_position - np.asarray(self.position)
            self.velocity = (
                self.velocity * cfg['hold_damping'] +
                displacement * cfg['hold_spring']
            )
            self.position = tuple(np.add(self.position, self.velocity * dt))
            previous_rotation = self.rotation
            self.rotation = self.rotation_smoother(
                raw_rot + self._grab_rotation_offset
            )
            self.depth_rotation = self.depth_smoother(
                raw_depth + self._grab_depth_offset
            )
            self.angular_velocity = self.rotation - previous_rotation
            self.length = self.length_smoother(raw_length)
            self._raw_position = raw_pos
            self._raw_rotation = raw_rot
            self._raw_depth_rotation = raw_depth
            return

        can_grab = grip_ratio <= cfg['max_grip_distance_ratio']
        close_to_cigarette = (
            self.position is not None and
            distance(raw_pos, self.position) <= cfg['grab_radius']
        )
        if can_grab and close_to_cigarette:
            self._begin_hold(raw_pos, raw_rot, raw_depth)
            self.length = self.length_smoother(raw_length)
        elif self.state == self.FALLING:
            self._update_falling(frame_shape, ashtray, dt)

    def get_tip_position(self):
        if self.position is None:
            return None
        projected_half_length = self.length * max(abs(np.cos(self.depth_rotation)), 0.2) / 2
        dx = np.cos(self.rotation) * projected_half_length
        dy = np.sin(self.rotation) * projected_half_length
        return (self.position[0] + dx, self.position[1] + dy)

    def get_base_position(self):
        if self.position is None:
            return None
        projected_half_length = self.length * max(abs(np.cos(self.depth_rotation)), 0.2) / 2
        dx = -np.cos(self.rotation) * projected_half_length
        dy = -np.sin(self.rotation) * projected_half_length
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

    def get_ember_position(self, mouth_center=None):
        """Return the cigarette end farthest from the mouth."""
        if self.position is None:
            return None

        tip = self.get_tip_position()
        base = self.get_base_position()
        if mouth_center is None:
            return tip

        tip_dist = distance(tip, mouth_center)
        base_dist = distance(base, mouth_center)
        return tip if tip_dist >= base_dist else base

    def get_render_rotation(self, mouth_center=None):
        """Orient the positive model axis toward the ember, away from the mouth."""
        if self.position is None or mouth_center is None:
            return self.rotation

        tip = self.get_tip_position()
        base = self.get_base_position()
        if distance(tip, mouth_center) < distance(base, mouth_center):
            return self.rotation + np.pi
        return self.rotation

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
            'depth_rotation': self.depth_rotation,
            'depth_rotation_deg': np.degrees(self.depth_rotation),
            'length': self.length,
            'state': self.state,
            'velocity': tuple(self.velocity),
            'frames_lost': self._frames_lost,
        }

    def reset(self):
        self.is_held = False
        self.position = None
        self.rotation = 0.0
        self.depth_rotation = Config.CIGARETTE_TRACKER['default_depth_rotation']
        self.length = self.base_length
        self._raw_position = None
        self._raw_rotation = 0.0
        self._raw_depth_rotation = Config.CIGARETTE_TRACKER['default_depth_rotation']
        self._last_valid_position = None
        self._last_valid_rotation = 0.0
        self._frames_lost = 0
        self.state = self.RESTING
        self.velocity[:] = 0.0
        self.angular_velocity = 0.0
        self._physics_initialized = False
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
        self.depth_smoother = AngleOneEuroFilter(
            freq=rot_cfg['freq'],
            mincutoff=rot_cfg['mincutoff'],
            beta=rot_cfg['beta']
        )
        self.length_smoother = OneEuroFilter(
            freq=pos_cfg['freq'],
            mincutoff=pos_cfg['mincutoff'],
            beta=pos_cfg['beta']
        )
