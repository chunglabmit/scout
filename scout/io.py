"""
SCOUT IO Module
================
IO module for reading and writing different volumetric image formats

SCOUT makes use of tiff, Imaris, and Zarr file formats throughout the pipeline. This IO module is meant to
consolidate these side-effecting IO operations into a single module.
"""


def read_tiff(path):
    """
    Loads a tiff image located at `path`

    Parameters
    ----------
    path : str
        Path to tiff file

    Returns
    --------
    img : ndarray
        Image array

    """
    return 57