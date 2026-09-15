import cv2
import numpy as np
import random
from config import Config


class SmokeParticle:
    """A soft parcel of exhaled smoke with drag, curl, and buoyancy."""

    def __init__(self, x, y, config, direction=(0.0, -0.12)):
        self.x = x
        self.y = y
        self.config = config

        direction_x, direction_y = direction
        direction_x *= config['direction_screen_gain']
        direction_length = max(1.0, np.hypot(direction_x, direction_y))
        direction_x /= direction_length
        direction_y /= direction_length
        spread = random.uniform(-config['direction_spread'], config['direction_spread'])
        speed = random.uniform(config['min_speed'], config['max_speed'])
        self.vx = (direction_x - direction_y * spread) * speed
        self.vy = (direction_y + direction_x * spread) * speed

        self.size = random.uniform(config['initial_size_min'], config['initial_size_max'])
        self.base_opacity = random.uniform(
            config['initial_opacity_min'], config['initial_opacity_max']
        )
        self.opacity = 0.0
        self.lifetime = random.uniform(config['lifetime_min'], config['lifetime_max'])
        self.age = 0.0

        self.expansion_rate = random.uniform(config['expansion_rate_min'], config['expansion_rate_max'])
        self.aspect_ratio = random.uniform(
            config['aspect_ratio_min'], config['aspect_ratio_max']
        )
        self.rotation = random.uniform(0.0, 180.0)
        self.rotation_speed = random.uniform(-0.45, 0.45)
        self.phase = random.uniform(0.0, np.pi * 2.0)
        self.turbulence = random.uniform(
            config['turbulence_strength_min'], config['turbulence_strength_max']
        )
        self.turbulence_frequency = random.uniform(
            config['turbulence_frequency_min'], config['turbulence_frequency_max']
        )

        self.color = (
            random.randint(config['color_r_min'], config['color_r_max']),
            random.randint(config['color_g_min'], config['color_g_max']),
            random.randint(config['color_b_min'], config['color_b_max'])
        )

    def update(self, dt=1.0):
        curl = np.sin(self.phase + self.age * self.turbulence_frequency)
        self.vx += curl * self.turbulence * dt
        self.vy += np.cos(
            self.phase * 0.7 + self.age * self.turbulence_frequency
        ) * self.turbulence * 0.3 * dt
        self.vy -= self.config['upward_force'] * (
            0.25 + self.age / self.lifetime
        ) * dt

        self.x += self.vx * dt
        self.y += self.vy * dt

        drag = self.config['velocity_drag'] ** dt
        self.vx *= drag
        self.vy *= drag

        self.size += self.expansion_rate * dt
        self.rotation += self.rotation_speed * dt

        self.age += dt
        life_progress = min(1.0, self.age / self.lifetime)
        fade_in = min(1.0, life_progress / 0.08)
        fade_out = max(0.0, 1.0 - life_progress) ** 1.65
        density_variation = 0.9 + 0.1 * np.sin(
            self.phase + self.age * self.turbulence_frequency * 0.6
        )
        self.opacity = self.base_opacity * fade_in * fade_out * density_variation

        return self.is_alive()

    def is_alive(self):
        return self.age < self.lifetime and self.opacity > 0.01 and self.size > 0.5

    def draw(self, frame):
        if not self.is_alive() or self.opacity <= 0:
            return

        int_x = int(self.x)
        int_y = int(self.y)
        int_size = max(1, int(self.size))

        h, w = frame.shape[:2]
        if int_x + int_size < 0 or int_x - int_size >= w or int_y + int_size < 0 or int_y - int_size >= h:
            return

        overlay = frame.copy()
        cv2.circle(overlay, (int_x, int_y), int_size, self.color, -1)

        alpha = min(1.0, self.opacity)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class InhaleSmokeParticle:
    """A small wisp that travels from the ember into the mouth."""

    def __init__(self, start, target, config):
        jitter = config['spawn_jitter']
        self.x = start[0] + random.uniform(-jitter, jitter)
        self.y = start[1] + random.uniform(-jitter, jitter)
        travel_frames = random.uniform(
            config['travel_frames_min'], config['travel_frames_max']
        )
        self.vx = (target[0] - self.x) / travel_frames
        self.vy = (target[1] - self.y) / travel_frames
        self.size = random.uniform(
            config['initial_size_min'], config['initial_size_max']
        )
        self.opacity = random.uniform(
            config['initial_opacity_min'], config['initial_opacity_max']
        )
        self.lifetime = travel_frames
        self.age = 0.0
        self.shrink_rate = config['shrink_rate']
        self.fade_rate = config['fade_rate']
        self.color = (205, 205, 205)
        self.start = np.asarray((self.x, self.y), dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)
        path = self.target - self.start
        path_length = max(1.0, float(np.hypot(path[0], path[1])))
        perpendicular = np.asarray((-path[1], path[0]), dtype=np.float32) / path_length
        curve_strength = random.uniform(
            config['curve_strength_min'], config['curve_strength_max']
        ) * random.choice((-1.0, 1.0))
        self.curve_offset = perpendicular * curve_strength
        self.aspect_ratio = random.uniform(0.65, 1.2)
        self.rotation = float(np.degrees(np.arctan2(path[1], path[0])))

    def update(self, dt=1.0):
        self.age += dt
        progress = min(1.0, self.age / self.lifetime)
        eased_progress = 1.0 - (1.0 - progress) ** 1.35
        position = (
            self.start * (1.0 - eased_progress) +
            self.target * eased_progress +
            self.curve_offset * np.sin(np.pi * progress)
        )
        self.x, self.y = float(position[0]), float(position[1])
        self.size = max(0.5, self.size * (1.0 - self.shrink_rate * dt))
        self.opacity = max(0.0, self.opacity * (1.0 - self.fade_rate * dt))
        return self.is_alive()

    def is_alive(self):
        return self.age < self.lifetime and self.opacity > 0.01

    def draw(self, frame):
        if not self.is_alive():
            return

        center = (int(self.x), int(self.y))
        radius = max(1, int(self.size))
        h, w = frame.shape[:2]
        if center[0] + radius < 0 or center[0] - radius >= w:
            return
        if center[1] + radius < 0 or center[1] - radius >= h:
            return

        overlay = frame.copy()
        cv2.circle(overlay, center, radius, (205, 205, 205), -1)
        cv2.addWeighted(overlay, min(1.0, self.opacity), frame, 1 - min(1.0, self.opacity), 0, frame)


class SmokeEffect:
    def __init__(self, config=None, inhale_config=None):
        self.config = config or self._default_config()
        self.inhale_config = inhale_config or Config.INHALE_SMOKE_EFFECT.copy()
        self.particles = []
        self.last_exhalation_state = False
        self.last_inhalation_state = False
        self.exhalation_triggered = False
        self._inhale_frame_count = 0
        self._exhale_frame_count = 0
        self._render_frame = 0

    def _default_config(self):
        return Config.SMOKE_EFFECT.copy()

    def update(self, exhalation_detected, mouth_center, dt=1.0,
               inhalation_detected=False, ember_position=None,
               exhale_direction=(0.0, -0.12)):
        if exhalation_detected:
            spawn_interval = self.config.get('spawn_interval_frames', 2)
            if not self.last_exhalation_state or self._exhale_frame_count >= spawn_interval:
                self._spawn_particles(mouth_center, exhale_direction)
                self._exhale_frame_count = 0
            self._exhale_frame_count += 1
            self.exhalation_triggered = True
        else:
            self.exhalation_triggered = False
            self._exhale_frame_count = 0

        self.last_exhalation_state = exhalation_detected

        if inhalation_detected and mouth_center is not None and ember_position is not None:
            spawn_interval = self.inhale_config['spawn_interval_frames']
            if not self.last_inhalation_state or self._inhale_frame_count >= spawn_interval:
                self._spawn_inhale_particles(ember_position, mouth_center)
                self._inhale_frame_count = 0
            self._inhale_frame_count += 1
        else:
            self._inhale_frame_count = 0
        self.last_inhalation_state = inhalation_detected

        alive_particles = []
        for p in self.particles:
            if p.update(dt):
                alive_particles.append(p)
        self.particles = alive_particles

        max_particles = self.config.get('max_particles', 90)
        if len(self.particles) > max_particles:
            self.particles = self.particles[-max_particles:]

    def _spawn_particles(self, mouth_center, direction=(0.0, -0.12)):
        if mouth_center is None:
            return

        offset_x = self.config.get('origin_offset_x', 0)
        offset_y = self.config.get('origin_offset_y', -8)

        count = random.randint(self.config['particle_count_min'], self.config['particle_count_max'])
        for _ in range(count):
            offset_x_rand = random.uniform(-8, 8)
            offset_y_rand = random.uniform(-4, 4)
            p = SmokeParticle(
                mouth_center[0] + offset_x + offset_x_rand,
                mouth_center[1] + offset_y + offset_y_rand,
                self.config,
                direction,
            )
            self.particles.append(p)

    def _spawn_inhale_particles(self, ember_position, mouth_center):
        count = random.randint(
            self.inhale_config['particle_count_min'],
            self.inhale_config['particle_count_max'],
        )
        for _ in range(count):
            self.particles.append(
                InhaleSmokeParticle(ember_position, mouth_center, self.inhale_config)
            )

    def draw(self, frame):
        if not self.particles:
            return

        height, width = frame.shape[:2]
        visible = []
        for particle in self.particles:
            if not particle.is_alive():
                continue

            x, y = int(particle.x), int(particle.y)
            radius_x = max(1, int(particle.size * particle.aspect_ratio))
            radius_y = max(1, int(particle.size / particle.aspect_ratio))
            radius = max(radius_x, radius_y)
            if x + radius < 0 or x - radius >= width:
                continue
            if y + radius < 0 or y - radius >= height:
                continue

            visible.append((particle, x, y, radius, radius_x, radius_y))

        if not visible:
            return

        sigma = self.config.get('blur_sigma', 3.5)
        blur_margin = max(2, int(sigma * 3))
        left = max(0, min(x - radius for _, x, _, radius, _, _ in visible) - blur_margin)
        top = max(0, min(y - radius for _, _, y, radius, _, _ in visible) - blur_margin)
        right = min(width, max(x + radius for _, x, _, radius, _, _ in visible) + blur_margin + 1)
        bottom = min(height, max(y + radius for _, _, y, radius, _, _ in visible) + blur_margin + 1)

        alpha_layer = np.zeros((bottom - top, right - left), dtype=np.float32)

        for particle, x, y, _, radius_x, radius_y in visible:
            particle_radius = max(radius_x, radius_y)
            local_left = max(0, x - left - particle_radius)
            local_top = max(0, y - top - particle_radius)
            local_right = min(right - left, x - left + particle_radius + 1)
            local_bottom = min(bottom - top, y - top + particle_radius + 1)
            particle_mask = np.zeros(
                (local_bottom - local_top, local_right - local_left),
                dtype=np.float32,
            )
            center = (x - left - local_left, y - top - local_top)
            angle = np.radians(particle.rotation)
            lobe_offset = int(particle_radius * 0.28)
            lobe_center = (
                center[0] + int(np.cos(angle) * lobe_offset),
                center[1] + int(np.sin(angle) * lobe_offset),
            )
            cv2.ellipse(
                particle_mask,
                lobe_center,
                (max(1, int(radius_x * 0.62)), max(1, int(radius_y * 0.62))),
                particle.rotation + 25.0, 0, 360,
                min(1.0, particle.opacity * 0.7), -1,
            )
            cv2.ellipse(
                particle_mask,
                center, (radius_x, radius_y), particle.rotation, 0, 360,
                min(1.0, particle.opacity), -1,
            )
            target = alpha_layer[
                local_top:local_bottom, local_left:local_right
            ]
            target[:] = 1.0 - (1.0 - target) * (1.0 - particle_mask)

        alpha_layer = cv2.GaussianBlur(alpha_layer, (0, 0), sigma)
        alpha = np.clip(alpha_layer, 0.0, 1.0)[..., None]
        texture_strength = self.config.get('density_texture_strength', 0.0)
        if texture_strength > 0:
            grid_y, grid_x = np.ogrid[top:bottom, left:right]
            phase = self._render_frame * 0.08
            texture = (
                1.0 - texture_strength * 0.5 +
                texture_strength * 0.28 * np.sin(grid_x * 0.055 + phase) +
                texture_strength * 0.22 * np.sin(
                    grid_y * 0.073 - grid_x * 0.031 - phase * 0.7
                )
            )
            alpha *= np.clip(texture, 0.55, 1.05)[..., None]
        self._render_frame += 1
        smoke_color = np.asarray((212, 212, 216), dtype=np.float32)
        frame_region = frame[top:bottom, left:right]
        frame_region[:] = (
            frame_region.astype(np.float32) * (1.0 - alpha) +
            smoke_color * alpha
        ).astype(np.uint8)

    def get_particle_count(self):
        return len(self.particles)

    def is_active(self):
        return len(self.particles) > 0

    def reset(self):
        self.particles = []
        self.last_exhalation_state = False
        self.last_inhalation_state = False
        self.exhalation_triggered = False
        self._inhale_frame_count = 0
        self._exhale_frame_count = 0
        self._render_frame = 0
