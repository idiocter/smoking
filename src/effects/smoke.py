import cv2
import numpy as np
import random
from config import Config


class SmokeParticle:
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.config = config

        angle_variation = random.uniform(-config['spread_angle'], config['spread_angle'])
        base_angle = config['base_angle'] + angle_variation
        speed = random.uniform(config['min_speed'], config['max_speed'])

        self.vx = np.cos(base_angle) * speed
        self.vy = np.sin(base_angle) * speed

        self.size = random.uniform(config['initial_size_min'], config['initial_size_max'])
        self.opacity = random.uniform(config['initial_opacity_min'], config['initial_opacity_max'])
        self.lifetime = random.uniform(config['lifetime_min'], config['lifetime_max'])
        self.age = 0.0

        self.expansion_rate = random.uniform(config['expansion_rate_min'], config['expansion_rate_max'])
        self.fade_rate = random.uniform(config['fade_rate_min'], config['fade_rate_max'])
        self.drift = random.uniform(-config['drift_strength'], config['drift_strength'])

        self.color = (
            random.randint(config['color_r_min'], config['color_r_max']),
            random.randint(config['color_g_min'], config['color_g_max']),
            random.randint(config['color_b_min'], config['color_b_max'])
        )

    def update(self, dt=1.0):
        self.x += self.vx * dt
        self.y += self.vy * dt

        self.vy -= self.config['upward_force'] * dt
        self.vx += self.drift * dt * 0.1

        self.size += self.expansion_rate * dt

        self.age += dt
        self.opacity = max(0.0, self.opacity * (1.0 - self.fade_rate * dt))

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

    def update(self, dt=1.0):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.size = max(0.5, self.size * (1.0 - self.shrink_rate * dt))
        self.opacity = max(0.0, self.opacity * (1.0 - self.fade_rate * dt))
        self.age += dt
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

    def _default_config(self):
        return Config.SMOKE_EFFECT.copy()

    def update(self, exhalation_detected, mouth_center, dt=1.0,
               inhalation_detected=False, ember_position=None):
        if exhalation_detected and not self.last_exhalation_state:
            self._spawn_particles(mouth_center)
            self.exhalation_triggered = True
        elif not exhalation_detected:
            self.exhalation_triggered = False

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

    def _spawn_particles(self, mouth_center):
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
                self.config
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
        for p in self.particles:
            p.draw(frame)

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
