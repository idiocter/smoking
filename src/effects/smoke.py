import numpy as np
import cv2
import random


class SmokeParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.5, 0.5)
        self.vy = random.uniform(-2.0, -0.5)
        self.size = random.uniform(3, 8)
        self.alpha = 1.0
        self.life = 1.0
        self.decay = random.uniform(0.01, 0.03)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy -= 0.02
        self.size += 0.15
        self.life -= self.decay
        self.alpha = max(0, self.life)

    def is_alive(self):
        return self.life > 0

    def draw(self, frame):
        if self.alpha <= 0:
            return
        color = (int(200 * self.alpha), int(200 * self.alpha), int(200 * self.alpha))
        cv2.circle(frame, (int(self.x), int(self.y)), int(self.size), color, -1)


class SmokeEffect:
    def __init__(self, max_particles=100):
        self.particles = []
        self.max_particles = max_particles
        self.emission_rate = 0
        self.active = False

    def set_emission(self, rate):
        self.emission_rate = rate
        self.active = rate > 0

    def update(self, mouth_pos):
        if self.active and mouth_pos and self.emission_rate > 0:
            for _ in range(int(self.emission_rate)):
                if len(self.particles) < self.max_particles:
                    offset_x = random.uniform(-10, 10)
                    offset_y = random.uniform(-5, 5)
                    self.particles.append(SmokeParticle(mouth_pos[0] + offset_x, mouth_pos[1] + offset_y))

        self.particles = [p for p in self.particles if p.is_alive()]
        for p in self.particles:
            p.update()

    def draw(self, frame):
        for p in self.particles:
            p.draw(frame)

    def clear(self):
        self.particles.clear()