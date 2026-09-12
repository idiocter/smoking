#!/usr/bin/env python3
"""Startup and command-line tests that do not access a webcam."""

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

import main
from camera.camera import Camera


def test_project_paths_are_independent_of_working_directory():
    previous_cwd = Path.cwd()
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chdir(temp_dir)
            assert main.validate_assets()
            assert main.FACE_MODEL_PATH.is_file()
            assert main.HAND_MODEL_PATH.is_file()
    finally:
        os.chdir(previous_cwd)


def test_missing_required_assets_are_reported():
    with tempfile.TemporaryDirectory() as temp_dir:
        assert not main.validate_assets(temp_dir)


def test_command_line_options():
    defaults = main.parse_args([])
    assert not defaults.debug
    assert defaults.prefer_3d

    options = main.parse_args(['--debug', '--2d'])
    assert options.debug
    assert not options.prefer_3d


def test_camera_accepts_configured_frame_rate():
    camera = Camera(fps=24)
    assert camera.target_fps == 24


if __name__ == '__main__':
    test_project_paths_are_independent_of_working_directory()
    test_missing_required_assets_are_reported()
    test_command_line_options()
    test_camera_accepts_configured_frame_rate()
    print('Startup tests passed')
