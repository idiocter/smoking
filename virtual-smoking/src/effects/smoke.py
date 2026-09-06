import cv2
import numpy as np
import random


class SmokeParticle:
    def __init__(self, x, y, config):
        self.x = x
        self.y = y
        self.config = config

        # Initial properties with small random variation
        angle_variation = random.uniform(-config['spread_angle'], config['spread_angle'])
        base_angle = config['base_angle'] + angle_variation
        speed = random.uniform(config['min_speed'], config['max_speed'])

        self.vx = np.cos(base_angle) * speed
        self.vy = np.sin(base_angle) * speed

        self.size = random.uniform(config['initial_size_min'], config['initial_size_max'])
        self.opacity = random.uniform(config['initial_opacity_min'], config['initial_opacity_max'])
        self.lifetime = random.uniform(config['lifetime_min'], config['lifetime_max'])
        self.age = 0.0

        # Expansion and fade rates
        self.expansion_rate = random.uniform(config['expansion_rate_min'], config['expansion_rate_max'])
        self.fade_rate = random.uniform(config['fade_rate_min'], config['fade_rate_max'])
        self.drift = random.uniform(-config['drift_strength'], config['drift_strength'])

        # Color (neutral smoke gray)
        self.color = (
            random.randint(config['color_r_min'], config['color_r_max']),
            random.randint(config['color_g_min'], config['color_g_max']),
            random.randint(config['color_b_min'], config['color_b_max'])
        )

    def update(self, dt=1.0):
        # Move
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Upward force (buoyancy)
        self.vy -= self.config['upward_force'] * dt

        # Horizontal drift
        self.vx += self.drift * dt * 0.1

        # Expand
        self.size += self.expansion_rate * dt

        # Age and fade
        self.age += dt
        life_progress = self.age / self.lifetime
        self.opacity = max(0.0, self.opacity * (1.0 - self.fade_rate * dt))

        return self.is_alive()

    def is_alive(self):
        return self.age < self.lifetime and self.opacity > 0.01 and self.size > 0.5

    def draw(self, frame):
        if not self.is_alive() or self.opacity <= 0:
            return

        # Create soft circular gradient
        int_x = int(self.x)
        int_y = int(self.y)
        int_size = max(1, int(self.size))

        # Bounds check
        h, w = frame.shape[:2]
        if int_x + int_size < 0 or int_x - int_size >= w or int_y + int_size < 0 or int_y - int_size >= h:
            return

        # Draw soft particle using multiple circles for gradient effect
        # Simpler approach: draw with cv2.circle with alpha blending
        overlay = frame.copy()
        cv2.circle(overlay, (int_x, int_y), int_size, self.color, -1)

        # Alpha blend
        alpha = min(1.0, self.opacity)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


class SmokeEffect:
    def __init__(self, config=None):
        self.config = config or self._default_config()
        self.particles = []
        self.last_exhalation_state = False
        self.exhalation_triggered = False

    def _default_config(self):
        return {
            # Spawning
            'particle_count_min': 8,
            'particle_count_max': 16,
            'spread_angle': 0.5,  # radians
            'base_angle': -np.pi / 2,  # upward (-pi/2 = straight up)

            # Initial properties
            'initial_size_min': 6,
            'initial_size_max': 14,
            'initial_opacity_min': 0.3,
            'initial_opacity_max': 0.5,
            'lifetime_min': 30,
            'lifetime_max': 60,

            # Physics
            'min_speed': 1.0,
            'max_speed': 3.0,
            'upward_force': 0.08,
            'drift_strength': 0.15,
            'expansion_rate_min': 0.15,
            'expansion_rate_max': 0.35,
            'fade_rate_min': 0.015,
            'fade_rate_max': 0.03,

            # Color
            'color_r_min': 160,
            'color_r_max': 200,
            'color_g_min': 160,
            'color_g_max': 200,
            'color_b_min': 160,
            'color_b_max': 200,
        }

    def update(self, exhalation_detected, mouth_center, dt=1.0):
        # Detect rising edge of exhalation (trigger once per exhalation)
        if exhalation_detected and not self.last_exhalation_state:
            self._spawn_particles(mouth_center)
            self.exhalation_triggered = True
        elif not exhalation_detected:
            self.exhalation_triggered = False

        self.last_exhalation_state = exhalation_detected

        # Update existing particles
        alive_particles = []
        for p in self.particles:
            if p.update(dt):
                alive_particles.append(p)
        self.particles = alive_particles

    def _spawn_particles(self, mouth_center):
        if mouth_center is None:
            return

        count = random.randint(self.config['particle_count_min'], self.config['particle_count_max'])
        for _ in range(count):
            # Small random offset from mouth center
            offset_x = random.uniform(-8, 8)
            offset_y = random.uniform(-4, 4)
            p = SmokeParticle(mouth_center[0] + offset_x, mouth_center[1] + offset_y, self.config)
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