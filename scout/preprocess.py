"""
Preprocess Module
==================

This module contains functions for image preprocessing before the main analysis.
These functions include gaussian smoothing, histogram equalization, and background elimination.

The `scout preprocess` command can be used to apply these functions as well as convert to a Zarr array.

"""

import multiprocessing
from functools import partial
import numpy as np
import tqdm
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist
from pybm3d.bm3d import bm3d
from scout.utils import extract_ghosted_chunk, extract_box, insert_box, pmap_chunks, verbose_print
from scout import io


def gaussian_blur(image, sigma):
    """
    Smooth input `image` by gaussian blurring with `sigma` kernel.

    This function does not normalize the image before applying the gaussian blurring.

    Parameters
    ----------
    image : ndarray
        Input image
    sigma : int, tuple
        Blurring amount for each axis

    Returns
    -------
    blurred : ndarray
        Smoothed image

    """
    return gaussian(image, sigma, preserve_range=True)


def _gaussian_blur_chunk(input_tuple, sigma, output, overlap):
    """
    Smooth a chunk from an input Zarr array

    Parameters
    ----------
    input_tuple : tuple
        Tuple containing a reference to a Zarr array, the chunk start coordinates, and the chunk size to process
    sigma : int, tuple
        Blurring amount for each axis
    output : Zarr Array
        Output Zarr array for results
    overlap : int
        Amount of pixel overlap between chunks to avoid edge effects

    """
    arr, start_coord, chunks = input_tuple
    ghosted_chunk, start_ghosted, stop_ghosted = extract_ghosted_chunk(arr,
                                                                       start_coord,
                                                                       chunks,
                                                                       overlap)
    g = gaussian_blur(ghosted_chunk, sigma)
    start_local = start_coord - start_ghosted
    stop_local = np.minimum(start_local + np.asarray(chunks),
                            np.asarray(ghosted_chunk.shape))
    g_valid = extract_box(g, start_local, stop_local)

    stop_coord = start_coord + np.asarray(g_valid.shape)
    insert_box(output, start_coord, stop_coord, g_valid)


def gaussian_blur_parallel(arr, sigma, output, chunks, overlap, nb_workers=None):
    """
    Smooths a chunked Zarr array with parallel processing.

    Parameters
    ----------
    arr : Zarr Array
    sigma : int, tuple
    output : Zarr Array
    chunks : tuple
    overlap : int
    nb_workers : int

    """
    f = partial(_gaussian_blur_chunk, sigma=sigma, output=output, overlap=overlap)
    pmap_chunks(f, arr, chunks, nb_workers=nb_workers, use_imap=True)


def clahe(image, kernel_size, clip_limit=0.01, nbins=256, nb_workers=None):
    """
    Apply CLAHE to each z-slice in `image`

    Parameters
    ----------
    image : ndarray
        input image
    kernel_size : int or list-like
        shape of the contextual regions
    clip_limit : float, optional
        limit for number of clipping pixels
    nbins : int, optional
        number of gray bins for histograms
    nb_workers : int, optional
        number of workers to use. Default, cpu_count

    Returns
    -------
    equalized : ndarray
        output image

    """
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    f = partial(equalize_adapthist,
                kernel_size=kernel_size,
                clip_limit=clip_limit,
                nbins=nbins)
    with multiprocessing.Pool(nb_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(f, image), total=image.shape[0]))
    image_min = image.min()
    image_max = image.max()
    output = np.asarray(results) * (image_max - image_min) + image_min
    return output.astype(image.dtype)


def remove_background(image, threshold):
    """Threshold an image and use as a mask to remove the background. Used for improved compression

    Parameters
    ----------
    image : ndarray
        input image
    threshold : float
        intensity to threshold the input image

    Returns
    -------
    filtered : ndarray
        output image with background set to 0

    """
    mask = (image >= threshold)
    return image * mask


def denoise2d(image, sigma, patch_size=0):
    """
    Denoise input `image` using BM3D collaborative filtering

    Parameters
    ----------
    image : ndarray
        Input 2D image
    sigma : float
        Noise standard deviation
    patch_size : int
        Size of patches to use in filtering. Default behavior, 0.

    Returns
    -------
    output : ndarray
        Denoised 2D image

    """
    output = bm3d(image, sigma, patch_size=patch_size, tau_2D_hard='BIOR', tau_2D_wien='BIOR')
    # Default useSD_h=True, useSD_w=True are used
    # Not sure why DCT-based filters are not working and giving severe block artifacts
    # Using bidirectional wavelets seems to be more stable
    return output


def denoise(image, sigma, patch_size=0, nb_workers=None):
    """
    Denoise input `image` slice-by-slice with BM3D collaborative filtering

    Parameters
    ----------
    image : ndarray
        Input 3D image
    sigma : float
        Noise standard deviation
    patch_size : int
        Patches to use in filtering. Default behavior, 0.
    nb_workers : int
        Number of parallel processes to use. Default, cpu_count()

    Returns
    -------
    output : ndarray
        Denoised 3D image

    """
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    f = partial(denoise2d, sigma=sigma, patch_size=patch_size)
    with multiprocessing.Pool(nb_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(f, image), total=image.shape[0]))
    return np.asarray(results)


# Define command-line functionality


def preprocess_cli(subparsers):
    preprocess_parser = subparsers.add_parser('preprocess', help="image preprocessing",
                                              description='Organoid image preprocessing tool')
    preprocess_parser.add_argument('image', help="Path to input TIFF image")
    preprocess_parser.add_argument('zarr', help="Path to output Zarr array")
    preprocess_parser.add_argument('-s', help="Standard deviation of noise", type=float, default=None)
    preprocess_parser.add_argument('-t', help="Threshold for background removal", type=float, default=None)
    preprocess_parser.add_argument('-k', help="CLAHE kernel size. (Use 0 for default)", type=int, default=None)
    preprocess_parser.add_argument('-c', help="Chunk size of output Zarr array", type=int, nargs='+', default=3*(64,))
    preprocess_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')
    return preprocess_parser


def preprocess_main(args):
    if args.t is None and args.s is None and args.k is None:
        raise ValueError('No preprocessing tasks were specified')

    # Load input image
    verbose_print(args, f"Preprocessing {args.image}")
    img = io.imread(args.image)
    shape, dtype = img.shape, img.dtype
    verbose_print(args, f"Loaded image: {shape} {dtype}")

    # Background removal
    if args.t is not None:
        verbose_print(args, f"Performing background removal with threshold {args.t}")
        img = remove_background(img, args.t)

    # Denoising
    if args.s is not None:
        verbose_print(args, f"Performing noise removal with sigma {args.s}")
        img = denoise(img, args.s)

    # Histogram equalization
    if args.k is not None:
        if args.k == 0:
            verbose_print(args, f"Performing histogram equalization with default kernel size")
            kernel_size = None
        else:
            verbose_print(args, f"Performing histogram equalization with kernel size {args.k}")
            kernel_size = args.k
        img = clahe(img, kernel_size=kernel_size)

    # Convert to Zarr
    verbose_print(args, f"Saving result to {args.zarr}")
    arr = io.new_zarr(args.zarr, shape=shape, dtype=dtype, chunks=tuple(args.c))
    arr[:] = img

    verbose_print(args, f"Preprocessing done!")
