#!/usr/bin/env python3
"""GLB parsing and transform tests that do not require an OpenGL context."""

import sys
from pathlib import Path

import numpy as np
from pygltflib import GLTF2
from pyrr import matrix44

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from effects.cigarette_3d import Cigarette3DRenderer


MODEL_PATH = Path(__file__).resolve().parents[1] / 'assets' / 'cigarette' / 'cigarette.glb'


class MockTracker:
    position = (320.0, 240.0)
    rotation = 0.0
    length = 140.0


def renderer_without_context():
    renderer = Cigarette3DRenderer.__new__(Cigarette3DRenderer)
    renderer.model_scale = 1.0
    renderer.model_offset = np.zeros(3, dtype=np.float32)
    renderer.rotation_offset = np.zeros(3, dtype=np.float32)
    renderer._model_extent = np.array([0.4628522, 0.04292809, 0.0429281], dtype=np.float32)
    return renderer


def screen_point(matrix, point, frame_height=720):
    world = matrix44.apply_to_vector(matrix, [*point, 1.0])
    return np.array([world[0], frame_height - world[1]])


def test_material_texture_slots():
    gltf = GLTF2().load(str(MODEL_PATH))
    sources = Cigarette3DRenderer._material_texture_sources(gltf)
    assert sources == {'albedo': 0, 'metallic_roughness': 1, 'emissive': 2}


def test_horizontal_transform_matches_tracker_length():
    renderer = renderer_without_context()
    tracker = MockTracker()
    matrix = renderer._model_matrix(720, tracker)
    half_extent = renderer._model_extent[0] / 2
    left = screen_point(matrix, (-half_extent, 0.0, 0.0))
    right = screen_point(matrix, (half_extent, 0.0, 0.0))

    assert np.allclose((left + right) / 2, tracker.position)
    assert np.isclose(np.linalg.norm(right - left), tracker.length)


def test_rotation_uses_image_coordinate_direction():
    renderer = renderer_without_context()
    tracker = MockTracker()
    tracker.rotation = np.pi / 2
    matrix = renderer._model_matrix(720, tracker)
    endpoint = screen_point(matrix, (renderer._model_extent[0] / 2, 0.0, 0.0))

    assert np.isclose(endpoint[0], tracker.position[0])
    assert endpoint[1] > tracker.position[1]


if __name__ == '__main__':
    test_material_texture_slots()
    test_horizontal_transform_matches_tracker_length()
    test_rotation_uses_image_coordinate_direction()
    print('3D cigarette tests passed')
