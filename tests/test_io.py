import pytest
from scout import io


def test_scout_io():
    assert io.read_tiff('path') == 57