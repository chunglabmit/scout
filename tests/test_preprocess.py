import subprocess
import os
import pytest
import numpy as np
from scout import io
from scout import preprocess
from skimage.transform import resize
import skimage.data
import matplotlib.pyplot as plt


@pytest.fixture
def image_path():
    return 'data/syto.tif'


@pytest.fixture
def image(image_path):
    return io.imread(image_path)


@pytest.fixture
def zarr_path(tmpdir):
    return os.path.join(tmpdir, 'example.zarr')


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


# preprocess CLI tests


def test_preprocess_threshold(image_path, image, zarr_path):
    threshold = 2000
    subprocess.call(['scout', 'preprocess', image_path, zarr_path, '-t', str(threshold)])
    data = io.open(zarr_path, mode='r')[:]
    assert np.isclose(data.min(), 0)
    assert np.isclose(data.max(), image.max())

# Denoising takes a long time

# def test_preprocess_denoise(image_path, image, zarr_path):
#     sigma = 20
#     subprocess.call(['scout', 'preprocess', image_path, zarr_path, '-s', str(sigma)])
#     data = io.open(zarr_path, mode='r')[:]
#     print(data.max(), image.max())
#     assert image.dtype == data.dtype
#     assert np.isclose(data.min(), image.min(), 0.1)
#     assert np.isclose(data.max(), image.max(), 0.1)


def test_preprocess_contrast(image_path, image, zarr_path):
    kernel_size = 63
    subprocess.call(['scout', 'preprocess', image_path, zarr_path, '-k', str(kernel_size)])
    data = io.open(zarr_path, mode='r')[:]
    assert data.dtype == image.dtype
    assert np.isclose(data.min(), image.min())
    assert np.isclose(data.max(), image.max())
    assert data.mean() > image.mean()
