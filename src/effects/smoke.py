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


class SmokeEffect:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.particles = []
        self.last_exhalation_state = False
        self.exhalation_triggered = False

    def _default_config(self):
        return Config.SMOKE_EFFECT.copy()

    def update(self, exhalation_detected, mouth_center, dt=1.0):
        if exhalation_detected and not self.last_exhalation_state:
            self._spawn_particles(mouth_center)
            self.exhalation_triggered = True
        elif not exhalation_detected:
            self.exhalation_triggered = False

        self.last_exhalation_state = exhalation_detected

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
        self.exhalation_triggered = False