import unittest
import numpy as np
import zarr
from functools import partial
from scout import utils


def double_elements(arr, start_coord, chunks):
    stop = np.minimum(start_coord + chunks, arr.shape)
    data = utils.extract_box(arr, start_coord, stop)
    return 2 * data


shm = utils.SharedMemory((25, 25), np.float32)


def double_elements_shm(arr, start_coord, chunks):
    stop = np.minimum(start_coord + chunks, arr.shape)
    with shm.txn() as arr:
        data = utils.extract_box(arr, start_coord, stop)
    return 2 * data


class TestPmapChunks(unittest.TestCase):

    def setUp(self):
        self.z = zarr.zeros((25, 25), chunks=(5, 5), dtype=np.float32)
        self.z[:] = np.arange(25**2).reshape((25, 25))
        self.f_zarr = double_elements

        self.shm = shm
        with shm.txn() as arr:
            arr[:] = self.z[:]
        self.f_shm = double_elements_shm

    def test_zarr(self):
        results = utils.pmap_chunks(self.f_zarr, self.z, nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())
        # using other shape chunks should still work
        results = utils.pmap_chunks(self.f_zarr, self.z, chunks=self.z.shape, nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())

    def test_shm(self):
        results = utils.pmap_chunks(self.f_shm, self.z, nb_workers=2)  # one chunk should still work
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())
        results = utils.pmap_chunks(self.f_shm, self.z, chunks=(5, 5), nb_workers=2)
        self.assertEqual(np.asarray(results).sum(), 2 * self.z[:].sum())
