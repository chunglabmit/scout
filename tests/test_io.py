import numpy as np
from scout import io


def test_imread():
    img = io.imread('tests/data/syto.tif')
    assert img.shape == (64, 128, 128)
    assert img.dtype == 'uint16'
    assert img.max() == 4095
    assert img.min() == 0
