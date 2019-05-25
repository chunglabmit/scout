"""
Preprocess Module
==================

This module contains functions for image preprocessing before the main analysis.
These functions include gaussian smoothing, histogram equalization, and background elimination.

The `scout preprocess` command can be used to apply these functions as well as convert to a Zarr array.

    - histogram : estimate image histogram
    - rescale : remove background below a threshold
    - denoise : remove noise using wavelet filtering
    - contrast : apply adaptive histrogram equalization (optional)
    - convert : convert image(s) to Zarr

"""

import os
import multiprocessing
import subprocess
from functools import partial
import numpy as np
import pandas as pd
import tqdm
from skimage.filters import gaussian
from skimage.exposure import equalize_adapthist, rescale_intensity, histogram
from skimage.restoration import denoise_wavelet, estimate_sigma
from skimage import img_as_float32
from scout.utils import extract_ghosted_chunk, extract_box, insert_box, pmap_chunks, verbose_print, tifs_in_dir
from scout import io
import matplotlib.pyplot as plt


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


def remove_background(image, threshold, offset=None):
    """Threshold an image and use as a mask to remove the background. May improve compression

    Parameters
    ----------
    image : ndarray
        input image
    threshold : float
        intensity to threshold the input image
    offset : float
        intensity to substract from the result

    Returns
    -------
    output : ndarray
        output image with background set to 0

    """
    mask = (image >= threshold)

    if offset is not None:
        output = np.clip(image * mask - offset, 0, None)
    else:
        output = image * mask

    return output


def downsample_paths(paths, step=100):
    return paths[0::step]


def estimate_histogram(paths, nbins=256):
    image = np.asarray([io.imread(path) for path in tqdm.tqdm(paths)])  # Load images into 3D array
    counts, bin_centers = histogram(image, nbins=nbins)
    return counts, bin_centers


def denoise2d(image, sigma, wavelet='db1'):
    """
    Denoise input `image` using wavelet filtering filtering

    Parameters
    ----------
    image : ndarray
        Input 2D image
    sigma : float
        Noise standard deviation
    wavelet : str
        Wavelet to use in DWT. Default, 'db1'.

    Returns
    -------
    output : ndarray
        Denoised 2D image

    """
    # sigma_estimate = estimate_sigma(image, average_sigmas=True)

    output = denoise_wavelet(image, sigma, wavelet=wavelet)
    # output = (output - output.min()) / (output.max() - output.min())  # Scale to [0, 1]
    # output = output * (image.max() - image.min()) + image.min()  # Scale to original range

    # Old PyBM3D code
    # output = bm3d(image, sigma, patch_size=patch_size, tau_2D_hard='BIOR', tau_2D_wien='BIOR')
    # # Default useSD_h=True, useSD_w=True are used
    # # Not sure why DCT-based filters are not working and giving severe block artifacts
    # # Using bidirectional wavelets seems to be more stable
    return output.astype(image.dtype)


def denoise(image, sigma, wavelet='db1', nb_workers=None):
    """
    Denoise input `image` slice-by-slice with wavelet filtering

    Parameters
    ----------
    image : ndarray
        Input 3D image
    sigma : float
        Noise standard deviation
    wavelet : str
        Wavelet to use in DWT. Default, 'db1'.
    nb_workers : int
        Number of parallel processes to use. Default, cpu_count()

    Returns
    -------
    output : ndarray
        Denoised 3D image

    """
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    f = partial(denoise2d, sigma=sigma, wavelet=wavelet)
    with multiprocessing.Pool(nb_workers) as pool:
        results = list(tqdm.tqdm(pool.imap(f, image), total=image.shape[0]))
    return np.asarray(results)


# Define command-line functionality

def process_write(img, f, output, compress=3):
    result = f(img)
    io.imsave(output, result, compress=compress)


def read_process_write(path, f, output, compress=3):
    # Get output path from input name and output folder
    filename = os.path.basename(path)
    output_path = os.path.join(output, filename)
    # Read image
    img = io.imread(path)
    # Process and write
    process_write(img, f, output_path, compress)


def preprocess_image3d(args, img):
    # Background removal
    if args.t is not None:
        verbose_print(args, f"Performing background removal with threshold {args.t}")
        img = remove_background(img, args.t)

    # Histogram equalization
    if args.k is not None:
        if args.k == 0:
            verbose_print(args, f"Performing histogram equalization with default kernel size")
            kernel_size = None
        else:
            verbose_print(args, f"Performing histogram equalization with kernel size {args.k}")
            kernel_size = args.k
        img = clahe(img, kernel_size=kernel_size)

    # Normalize and convert to float
    if args.float:
        img = rescale_intensity(img_as_float32(img))
        verbose_print(args, f"Converted to normalized float32: min {img.min():.3f}, max {img.max():.3f}")

    # Denoising
    if args.s is not None:
        verbose_print(args, f"Performing noise removal with sigma {args.s} and wavelet {args.w}")
        img = denoise(img, args.s, args.w)

    # Convert to Zarr
    verbose_print(args, f"Saving result to {args.zarr}")
    arr = io.new_zarr(args.zarr, shape=img.shape, dtype=img.dtype, chunks=tuple(args.c))
    arr[:] = img

    return img


def preprocess_image2d(args, path, arr, i):
    img = io.imread(path)

    # Background removal
    if args.t is not None:
        # verbose_print(args, f"Performing background removal with threshold {args.t}")
        img = remove_background(img, args.t)

    # Histogram equalization
    if args.k is not None:
        if args.k == 0:
            # verbose_print(args, f"Performing histogram equalization with default kernel size")
            kernel_size = None
        else:
            # verbose_print(args, f"Performing histogram equalization with kernel size {args.k}")
            kernel_size = args.k
        img = equalize_adapthist(img, kernel_size=kernel_size)

    # Convert to float (can't normalize based on single slice)
    if args.float:
        img = img_as_float32(img)
        # verbose_print(args, f"Converted to normalized float32: min {img.min():.3f}, max {img.max():.3f}")

    # Denoising
    if args.s is not None:
        # verbose_print(args, f"Performing noise removal with sigma {args.s} and wavelet {args.w}")
        img = denoise2d(img, args.s, args.w)

    arr[i] = img


def _preprocess_image2d(inputs):
    preprocess_image2d(*inputs)


def old_preprocessing_main(args):
    if args.t is None and args.s is None and args.k is None:
        raise ValueError('No preprocessing tasks were specified')

    verbose_print(args, f"Preprocessing {args.input}")

    if os.path.isdir(args.input):
        # Load series of 2D TIFFs and process in parallel
        paths, filenames = tifs_in_dir(args.input)

        img = io.imread(paths[0])
        shape = (len(paths), *img.shape)
        if args.float:
            dtype = 'float32'
        else:
            dtype = img.dtype

        arr = io.new_zarr(args.zarr, shape=shape, dtype=dtype, chunks=tuple(args.c))

        args_list = []
        for i, (path, _) in enumerate(zip(paths, filenames)):
            args_list.append((args, path, arr, i))

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            list(tqdm.tqdm(pool.imap_unordered(_preprocess_image2d, args_list), total=len(args_list)))

        if args.p is not None:
            before = io.imread(paths[args.p])
            after = arr[args.p]

    elif os.path.isdir(args.input):
        # Load 3D TIFF and process in memory
        img = io.imread(args.input)
        # Keep reference to before image if plotting
        if args.p is not None:
            before = np.copy(img[args.p])
        verbose_print(args, f"Loaded image: {img.shape} {img.dtype}")
        img = preprocess_image3d(args, img)
        if args.p is not None:
            after = np.copy(img[args.p])

    else:
        raise ValueError('Input is not a valid directory or file')

    # Show A/B plot
    if args.p is not None:
        plt.subplot(121)
        plt.imshow(before)
        plt.title('Before')
        plt.subplot(122)
        plt.imshow(after)
        plt.title('After')
        plt.show()

    verbose_print(args, f"Preprocessing done!")


# Newer interface ##############################

def _check_input(args):
    if os.path.isdir(args.input):
        verbose_print(args, f"Preprocessing 2D TIFFs in {args.input}")
        is_folder = True
    elif os.path.isfile(args.input):
        verbose_print(args, f"Preprocessing 3D TIFF {args.input}")
        is_folder = False
    else:
        raise ValueError('Input is not a valid directory or file')
    return is_folder


def _check_workers(args):
    if args.n is None:
        nb_workers = multiprocessing.cpu_count()
    else:
        nb_workers = args.n
    return nb_workers


def histogram_cli(subparsers):
    rescale_parser = subparsers.add_parser('histogram', help="Estimate histogram",
                                           description='Estimate image histogram quickly by skipping images')
    rescale_parser.add_argument('input', help="Path to folder of 2D TIFFs")
    rescale_parser.add_argument('output', help="Path CSV of histogram")
    rescale_parser.add_argument('-s', help="Step size for image sampling", type=int, default=50)
    rescale_parser.add_argument('-p', '--plot', help="Plot flag", action='store_true')
    rescale_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def histogram_main(args):
    # Find all TIFFs
    paths, _ = tifs_in_dir(args.input)
    verbose_print(args, f"Found {len(paths)} TIFFs")

    # Estimate histogram
    sample_paths = downsample_paths(paths, step=args.s)
    verbose_print(args, f"Calculating histogram from {len(sample_paths)} images:")
    hist, bin_centers = estimate_histogram(sample_paths)

    # Show plot
    if args.plot:
        plt.plot(bin_centers, hist)
        plt.show()

    # Build CSV
    df = pd.DataFrame({'intensity': bin_centers, 'count': hist})
    df.to_csv(args.output, index=False)
    verbose_print(args, f"Histogram saved to {args.output}")

    verbose_print(args, f"Histogram done!")


def rescale_cli(subparsers):
    rescale_parser = subparsers.add_parser('rescale', help="Remove background",
                                           description='Remove image background by thresholding')
    rescale_parser.add_argument('input', help="Path to folder of 2D TIFFs")
    rescale_parser.add_argument('histogram', help="Path to CSV of histogram")
    rescale_parser.add_argument('output', help="Path to output folder")
    rescale_parser.add_argument('-t', help="Threshold for removing background", type=float, default=0)
    rescale_parser.add_argument('-n', help="Number of workers. Default, cpu_count", type=int, default=None)
    rescale_parser.add_argument('-c', help="Output TIFF compression (0-9)", type=int, default=3)
    rescale_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def rescale_image(path, threshold, max_val, output, filename, compress):
    img = io.imread(path).astype(np.float32)  # load image as float

    # Subtract threshold and remove background by clipping negative values
    img -= threshold
    img = np.clip(img, 0, None)

    # Divide by max_val (accounting for threshold) to scale to [0, 1]
    img = img / (max_val - threshold)

    # Save result
    output_path = os.path.join(output, filename)
    io.imsave(output_path, img, compress=compress)


def _rescale_image(inputs):
    rescale_image(*inputs)


def rescale_main(args):
    nb_workers = _check_workers(args)

    # Find all TIFFs
    paths, filenames = tifs_in_dir(args.input)
    verbose_print(args, f"Found {len(paths)} TIFFs")

    # Load histogram
    df = pd.read_csv(args.histogram)
    bins = list(df['intensity'])
    min_val, max_val = bins[0], bins[-1]

    # Make the output folder
    os.makedirs(args.output, exist_ok=True)

    # Rescale images in parallel
    verbose_print(args, f"Rescaling images with {nb_workers} workers:")
    args_list = []
    for path, filename in zip(paths, filenames):
        args_list.append((path, args.t, max_val, args.output, filename, args.c))
    with multiprocessing.Pool(nb_workers) as pool:
        list(tqdm.tqdm(pool.imap(_rescale_image, args_list), total=len(paths)))

    verbose_print(args, f"Rescaling done!")


def denoise_cli(subparsers):
    denoise_parser = subparsers.add_parser('denoise', help="Denoise images",
                                           description='Denoise images using wavelet filtering')
    denoise_parser.add_argument('input', help="Path to folder of 2D TIFFs")
    denoise_parser.add_argument('output', help="Path to output folder")
    denoise_parser.add_argument('-s', help="Noise standard deviation. Default, estimated", type=float, default=None)
    denoise_parser.add_argument('-w', help="Wavelet for decomposition. Default, db1", default='db1')
    denoise_parser.add_argument('-n', help="Number of workers. Default, cpu_count", type=int, default=None)
    denoise_parser.add_argument('-c', help="Output TIFF compression (0-9)", type=int, default=3)
    denoise_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def denoise_main(args):
    # Initial setup
    nb_workers = _check_workers(args)
    os.makedirs(args.output, exist_ok=True)

    # Find all TIFFs
    paths, _ = tifs_in_dir(args.input)
    verbose_print(args, f"Found {len(paths)} TIFFs")

    # Curry denoising function for pmap
    f = partial(denoise2d, sigma=args.s, wavelet=args.w)
    g = partial(read_process_write, f=f, output=args.output, compress=args.c)

    # Parallel read, denoise, write
    verbose_print(args, f"Denoising with {nb_workers} workers:")
    with multiprocessing.Pool(nb_workers) as pool:
        list(tqdm.tqdm(pool.imap(g, paths), total=len(paths)))

    verbose_print(args, f"Denoising done!")


def contrast_cli(subparsers):
    contrast_parser = subparsers.add_parser('contrast', help="Adjust contrast",
                                            description='Adjust contrast using adaptive histogram equalization')
    contrast_parser.add_argument('input', help="Path to 3D TIFF or folder of 2D TIFFs")
    contrast_parser.add_argument('output', help="Path to output folder")
    contrast_parser.add_argument('-k', help="Kernel size", type=int, default=None)
    contrast_parser.add_argument('-c', help="Output TIFF compression (0-9)", type=int, default=3)
    contrast_parser.add_argument('-n', help="Number of workers. Default, cpu_count", type=int, default=None)
    contrast_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def contrast_main(args):
    # Initial setup
    nb_workers = _check_workers(args)

    if args.k is None:
        verbose_print(args, f"Performing histogram equalization with default kernel size")
        kernel_size = None
    else:
        verbose_print(args, f"Performing histogram equalization with kernel size {args.k}")
        kernel_size = args.k

    # Find all TIFFs
    paths, filenames = tifs_in_dir(args.input)
    verbose_print(args, f"Found {len(paths)} TIFFs")

    # Make output folder
    os.makedirs(args.output, exist_ok=True)

    for path, filename in tqdm.tqdm(zip(paths, filenames), total=len(paths)):
        img = io.imread(path)
        adjusted = equalize_adapthist(img, kernel_size=kernel_size).astype(np.float32)
        io.imsave(os.path.join(args.output, filename), adjusted, compress=args.c)

    verbose_print(args, f"Contrast done!")


def convert_cli(subparsers):
    convert_parser = subparsers.add_parser('convert', help="Convert TIFF to Zarr",
                                           description='Convert TIFF to Zarr NestedDirectoryStore')
    convert_parser.add_argument('input', help="Path to 3D TIFF or folder of 2D TIFFs")
    convert_parser.add_argument('output', help="Path to output Zarr array")
    convert_parser.add_argument('-c', help="Chunk size for Zarr output", type=int, nargs='+', default=3*(64,))
    convert_parser.add_argument('-n', help="Number of workers", type=int, default=None)
    convert_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def convert_batch(paths_batch, i, size, arr):
    start = i * size
    stop = start + len(paths_batch)
    img = np.asarray([io.imread(path) for path in paths_batch])
    arr[start:stop] = img


def _convert_batch(inputs):
    convert_batch(*inputs)


def convert_main(args):
    nb_workers = _check_workers(args)

    verbose_print(args, f"Converting {args.input} to Zarr")

    # Find all TIFFs
    paths, filenames = tifs_in_dir(args.input)
    verbose_print(args, f"Found {len(paths)} TIFFs")
    paths_chunked = [paths[pos:pos + args.c[0]] for pos in range(0, len(paths), args.c[0])]

    img = io.imread(paths[0])
    shape = (len(paths), *img.shape)
    dtype = img.dtype
    chunks = tuple(args.c)
    arr = io.new_zarr(args.output, shape=shape, dtype=dtype, chunks=chunks)

    verbose_print(args, f"Writiing to {args.output}")
    args_list = []
    for i, paths_batch in enumerate(paths_chunked):
        args_list.append((paths_batch, i, chunks[0], arr))
    with multiprocessing.Pool(nb_workers) as pool:
        list(tqdm.tqdm(pool.imap(_convert_batch, args_list), total=len(args_list)))

    verbose_print(args, f"Conversion done!")


def preprocess_cli(subparsers):
    preprocess_parser = subparsers.add_parser('preprocess', help="Image preprocessing tool",
                                              description='Organoid image preprocessing tool')
    preprocess_subparsers = preprocess_parser.add_subparsers(dest='preprocess_command',
                                                             title='preprocessing subcommands')
    histogram_cli(preprocess_subparsers)
    rescale_cli(preprocess_subparsers)
    denoise_cli(preprocess_subparsers)
    contrast_cli(preprocess_subparsers)
    convert_cli(preprocess_subparsers)
    return preprocess_subparsers


def preprocess_main(args):
    commands_dict = {
        'histogram': histogram_main,
        'rescale': rescale_main,
        'denoise': denoise_main,
        'contrast': contrast_main,
        'convert': convert_main,
    }
    func = commands_dict.get(args.preprocess_command, None)
    if func is None:
        print("Pickle Rick uses preprocessing subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'preprocess', '-h'])
    else:
        func(args)
