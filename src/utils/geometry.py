import math
import numpy as np


def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def angle_between(p1, p2, p3):
    v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
    v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
    dot = np.dot(v1, v2)
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    if norm == 0:
        return 0.0
    return math.acos(np.clip(dot / norm, -1.0, 1.0))


def midpoint(p1, p2):
    return ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)


def vector_angle(v):
    return math.atan2(v[1], v[0])


def normalize(v):
    norm = math.hypot(v[0], v[1])
    if norm == 0:
        return (0.0, 0.0)
    return (v[0] / norm, v[1] / norm)


def rotate_point(point, center, angle):
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    x = point[0] - center[0]
    y = point[1] - center[1]
    rx = x * cos_a - y * sin_a
    ry = x * sin_a + y * cos_a
    return (rx + center[0], ry + center[1])


def clamp(value, min_val, max_val):
    return max(min_val, min(max_val, value))