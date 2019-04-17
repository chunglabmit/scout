import numpy as np
from scout import io


def test_imread():
    img = io.imread('data/syto.tif')
    assert img.shape == (64, 1024, 1024)
    assert img.dtype == 'uint16'
    assert img.max() == 4095
    assert img.min() == 0
