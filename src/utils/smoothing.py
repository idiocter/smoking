from collections import deque
import math


class Smoother:
    def __init__(self, window_size=5):
        self.window_size = window_size
        self.values = deque(maxlen=window_size)

    def add(self, value):
        self.values.append(value)
        return self.get()

    def get(self):
        if not self.values:
            return None
        if isinstance(self.values[0], tuple):
            xs = [v[0] for v in self.values]
            ys = [v[1] for v in self.values]
            return (sum(xs) / len(xs), sum(ys) / len(ys))
        return sum(self.values) / len(self.values)

    def clear(self):
        self.values.clear()


class OneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.x_prev = None
        self.dx_prev = None

    def __call__(self, x):
        if self.x_prev is None:
            self.x_prev = x
            self.dx_prev = 0.0
            return x

        alpha = self._alpha(self.mincutoff)
        dx = (x - self.x_prev) * self.freq
        dx_hat = self._alpha(self.dcutoff) * dx + (1 - self._alpha(self.dcutoff)) * self.dx_prev
        cutoff = self.mincutoff + self.beta * abs(dx_hat)
        alpha = self._alpha(cutoff)
        x_hat = alpha * x + (1 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        return x_hat

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.x_prev = None
        self.dx_prev = None


class AngleOneEuroFilter:
    def __init__(self, freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.angle_prev = None
        self.d_angle_prev = None

    def __call__(self, angle):
        if self.angle_prev is None:
            self.angle_prev = angle
            self.d_angle_prev = 0.0
            return angle

        # Compute shortest angular difference
        diff = angle - self.angle_prev
        while diff > math.pi:
            diff -= 2 * math.pi
        while diff < -math.pi:
            diff += 2 * math.pi

        alpha = self._alpha(self.mincutoff)
        d_angle = diff * self.freq
        d_angle_hat = self._alpha(self.dcutoff) * d_angle + (1 - self._alpha(self.dcutoff)) * self.d_angle_prev
        cutoff = self.mincutoff + self.beta * abs(d_angle_hat)
        alpha = self._alpha(cutoff)

        # Apply filtered angular change
        angle_hat = self.angle_prev + alpha * diff

        # Normalize result
        while angle_hat > math.pi:
            angle_hat -= 2 * math.pi
        while angle_hat < -math.pi:
            angle_hat += 2 * math.pi

        self.angle_prev = angle_hat
        self.d_angle_prev = d_angle_hat
        return angle_hat

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def reset(self):
        self.angle_prev = None
        self.d_angle_prev = None