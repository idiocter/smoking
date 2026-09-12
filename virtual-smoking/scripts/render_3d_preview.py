#!/usr/bin/env python3
"""Render the bundled cigarette GLB to a PNG without using the webcam."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from effects.cigarette_3d import Cigarette3DRenderer


class PreviewTracker:
    is_held = True

    def __init__(self, position, rotation, length):
        self.position = position
        self.rotation = rotation
        self.length = length


def render_preview(output_path, width=960, height=540):
    x_gradient = np.linspace(42, 20, width, dtype=np.uint8)
    background = np.repeat(x_gradient[np.newaxis, :], height, axis=0)
    frame = np.dstack((background, background, background))

    tracker = PreviewTracker(
        position=(width / 2, height / 2),
        rotation=np.deg2rad(-12),
        length=min(width * 0.58, 560),
    )
    renderer = Cigarette3DRenderer(PROJECT_ROOT / 'assets' / 'cigarette' / 'cigarette.glb')
    try:
        renderer.set_view_projection(width, height)
        renderer.update_glow(True)
        for _ in range(7):
            preview = renderer.render(frame.copy(), tracker)
    finally:
        renderer.close()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), preview):
        raise RuntimeError(f'Could not write preview to {output_path}')
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'output' / 'cigarette_3d_preview.png',
        help='destination PNG path',
    )
    args = parser.parse_args(argv)

    try:
        output_path = render_preview(args.output)
    except Exception as exc:
        parser.exit(1, f'Preview rendering failed: {exc}\n')
    print(f'Preview written to {output_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
