import pytest
import numpy as np
from scout import io
from scout import preprocess
from skimage.transform import resize
import skimage.data
import matplotlib.pyplot as plt


@pytest.fixture
def image():
    return io.imread('tests/example.tif')


def test_denoise2d(image):
    img = image[0]
    output = preprocess.denoise2d(img, sigma=200)
    assert img.shape == output.shape
    assert img.dtype == output.dtype


def test_denoise(image):
    output = preprocess.denoise(image, sigma=200)
    assert image.shape == output.shape
    assert image.dtype == output.dtype
