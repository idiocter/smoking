import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from utils.geometry import distance, midpoint, angle_between, normalize, clamp
from utils.smoothing import Smoother, OneEuroFilter, AngleOneEuroFilter


def test_geometry():
    assert distance((0, 0), (3, 4)) == 5.0
    assert midpoint((0, 0), (10, 10)) == (5.0, 5.0)
    assert normalize((3, 4)) == (0.6, 0.8)
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    print("Geometry tests passed")


def test_smoothing():
    s = Smoother(window_size=3)
    s.add((0, 0))
    s.add((10, 10))
    s.add((20, 20))
    assert s.get() == (10.0, 10.0)
    s.add((30, 30))
    assert s.get() == (20.0, 20.0)
    print("Smoothing tests passed")


def test_one_euro_filter():
    f = OneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0)
    # First value passes through
    assert f(10.0) == 10.0
    # Second value should be smoothed
    result = f(12.0)
    assert 10.0 < result < 12.0
    print("OneEuroFilter tests passed")


def test_angle_one_euro_filter():
    f = AngleOneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0)
    # First value passes through
    assert f(0.0) == 0.0
    # Small angle change
    result = f(0.1)
    assert 0.0 < result < 0.1
    # Test angle wrapping - 179 to -179 should be small change
    f2 = AngleOneEuroFilter(freq=30.0, mincutoff=1.0, beta=0.0, dcutoff=1.0)
    f2(3.1)  # ~179 deg
    result = f2(-3.1)  # ~-179 deg, should be near 3.1 not -3.1
    assert result > 0  # Should not wrap the wrong way
    print("AngleOneEuroFilter tests passed")


if __name__ == '__main__':
    test_geometry()
    test_smoothing()
    test_one_euro_filter()
    test_angle_one_euro_filter()
    print("All tests passed!")