import os
import pytest
import numpy as np
from scout import io


@pytest.fixture
def image_path():
    return 'data/syto.tif'


@pytest.fixture
def image(image_path):
    return io.imread(image_path)


@pytest.fixture
def zarr_path(tmpdir):
    return os.path.join(tmpdir, 'example.zarr')



