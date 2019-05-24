"""
IO Module
===========
IO module for reading and writing different volumetric image formats

SCOUT makes use of tiff and Zarr file formats throughout the pipeline.
This IO module is meant to consolidate these side-effecting IO operations into a single module.
"""
import tifffile
import multiprocessing
import numpy as np
import zarr
from tqdm import tqdm
from numcodecs import Blosc
from scout.utils import tifs_in_dir


def imread(path):
    """
    Reads TIFF file into a numpy array in memory.

    Parameters
    -----------
    path : str
        Path to tif image to open

    Returns
    --------
    image : ndarray
        Image array
    """
    return tifffile.imread(files=path)


def imsave(path, data, compress=1):
    """
    Saves numpy array as a TIFF image.

    Parameters
    -----------
    path : str
        Path to tif image to create / overwrite
    data : ndarray
        Image data array
    compress : int
        Level of lossless TIFF compression (0-9)
    """
    tifffile.imsave(file=path, data=data, compress=compress)


def imread_parallel(paths, nb_workers):
    """
    Reads TIFF files into a numpy array in memory.

    Parameters
    ------------
    paths : list
        A list of TIFF paths to read (order is preserved)
    nb_workers : int
        Number of parallel processes to use in reading images

    Returns
    --------
    data : ndarray
        Image data
    """
    img = imread(paths[0])
    with multiprocessing.Pool(nb_workers) as pool:
        data = list(tqdm(pool.imap(imread, paths), total=len(paths)))
    return np.asarray(data, dtype=img.dtype)


def imread_folder(path, nb_workers):
    """
    Finds all TIFF images in a folder and loads them into a single array.

    Note that all images must be the same size to be able to stack them.

    Parameters
    ----------
    path : str
        Path to directory with TIFF images in alphabetical order
    nb_workers : int
        Number of parallel processes to use in reading images

    Returns
    -------
    data : ndarray
        Image array
    """
    paths, _ = tifs_in_dir(path)
    data = imread_parallel(paths, nb_workers)
    return data


def open(path, nested=True, mode='a'):
    """
    Opens a persistent Zarr array or NestedDirectoryStore located at `path`.

    Parameters
    ----------
    path : str
        Path to Zarr array or NestedDirectoryStore
    nested : bool
        Flag to indicate if path is for flat Zarr array or NestedDirectoryStore
    mode : str
        Read / write permissions mode

    Returns
    -------
    arr : zarr Array
        Reference to open Zarr array
    """
    if nested:
        store = zarr.NestedDirectoryStore(path)
        return zarr.open(store, mode=mode)
    else:
        return zarr.open(path, mode=mode)


def new_zarr(path, shape, chunks, dtype, in_memory=False, **kwargs):
    """
    Create new Zarr NestedDirectoryStore at `path`

    Parameters
    ----------
    path : str
        Path to new zarr array
    shape : tuple
        Overall shape of the zarr array
    chunks : tuple
        Shape of each chunk for the zarr array
    dtype : str
        Data type of for the zarr array
    kwargs : dict
        Keyword args to passs to zarr.open()

    Returns
    -------
    arr : zarr Array
        Reference to open zarr array
    """
    compressor = Blosc(cname='zstd', clevel=1, shuffle=Blosc.BITSHUFFLE)
    if in_memory:
        z_arr_out = zarr.zeros(shape=shape,
                               chunks=chunks,
                               dtype=dtype,
                               compressor=compressor,
                               **kwargs)
    else:
        store = zarr.NestedDirectoryStore(path)
        z_arr_out = zarr.open(store,
                              mode='w',
                              shape=shape,
                              chunks=chunks,
                              dtype=dtype,
                              compressor=compressor,
                              **kwargs)
    return z_arr_out


def new_zarr_like(path, arr, **kwargs):
    """
    Creates a new zarr array like `arr`.

    Parameters
    ----------
    path : str
        Path to new zarr array
    arr : zarr Array
        Reference to template zarr array
    kwargs : dict
        Keyword args to passs to zarr.open()

    Returns
    -------
    new_arr : zarr Array
        Reference to new zarr array
    """
    return new_zarr(path, arr.shape, arr.chunks, arr.dtype, **kwargs)
