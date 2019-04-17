import pytest
import numpy as np
from scout import io
from scout import preprocess
from skimage.transform import resize
import skimage.data
import matplotlib.pyplot as plt


@pytest.fixture
def image():
    return io.imread('data/syto.tif')


def test_denoise2d(image):
    img = image[0]
    img = img[:128, :128]
    output = preprocess.denoise2d(img, sigma=50)
    assert img.shape == output.shape
    assert img.dtype == output.dtype


def test_denoise(image):
    image = image[:8, :128, :128]
    output = preprocess.denoise(image, sigma=50)
    assert image.shape == output.shape
    assert image.dtype == output.dtype
