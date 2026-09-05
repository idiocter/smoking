import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.geometry import distance, midpoint, angle_between, normalize, clamp
from utils.smoothing import Smoother


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


if __name__ == '__main__':
    test_geometry()
    test_smoothing()
    print("All tests passed!")