"""
Segment Module
===============

This module performs organoid segmentation

These include the following subcommands:

    - downsample : reduce image size for faster segmentaion
    - ventricle : segment ventricles from downsampled nuclei image
    - foreground : segment organoid foreground
    - smooth : smooth a segmentation

"""

import os
import subprocess
import multiprocessing
import warnings
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from skimage.transform import downscale_local_mean
from skimage.segmentation import clear_border
from scipy.ndimage.morphology import binary_fill_holes
import torch
from scout import io
from scout.preprocess import gaussian_blur
from scout import utils
from scout.utils import verbose_print
from scout.unet_model import UNet


# Downsample Zarr

def downsample(arr, factors):
    """
    Downsample image `arr` by factors in each dimension

    Parameters
    ----------
    arr : Zarr Array
    factors : tuple

    Returns
    -------
    output : ndarray

    """
    if not isinstance(arr, np.ndarray):
        # Load image
        data = arr[:]
    else:
        data = arr
    # Resize image
    output = downscale_local_mean(data, factors).astype(arr.dtype)
    return output


# Ventricle segmentation

def load_model(path, device):
    model = UNet(1, 1)
    model.load_state_dict(torch.load(path))
    model.to(device)
    # model.half()  # Convert weights to 16-bit floats for less GPU memory usage
    return model


def segment_ventricles(model, data, t, device):
    output = np.empty(data.shape, dtype=np.float32)
    for i, img in tqdm(enumerate(data), total=len(data)):
        img = img.astype(np.float32)[np.newaxis, np.newaxis]
        img_tensor = torch.from_numpy(img).to(device)
        with warnings.catch_warnings():  # Suppress deprecated warning for Upsampling
            warnings.simplefilter('ignore')
            prob_tensor = model(img_tensor)
        prob = prob_tensor.detach().cpu().numpy()[0, 0]
        binary = (prob > t).astype(np.uint8)
        output[i] = binary
    return output


# Calculate local densities and threshold


# Segment niches (unused)


# rasterized = None
#
#
# def _rasterize_chunk(args):
#     start, shape, chunks, pts, labels = args
#     global rasterized
#     stop = np.minimum(shape, start + np.asarray(chunks))
#     grid_z, grid_y, grid_x = np.mgrid[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
#     data = griddata(pts, labels, (grid_z, grid_y, grid_x), method='nearest').astype(np.uint8)
#     rasterized = utils.insert_box(rasterized, start, stop, data)
#
#
# def rasterize_regions(pts, labels, shape, chunks=None, nb_workers=None):
#     global rasterized
#     if nb_workers is None:
#         nb_workers = multiprocessing.cpu_count()
#     if chunks is None:
#         grid_z, grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
#         rasterized = griddata(pts, labels, (grid_z, grid_y, grid_x), method='nearest').astype(np.uint8)
#     else:
#         chunk_coords = utils.chunk_coordinates(shape, chunks)
#         args_list = []
#         for start in tqdm(chunk_coords, total=len(chunk_coords)):
#             args_list.append((start, shape, chunks, pts, labels))
#         rasterized = utils.SharedMemory(shape=shape, dtype=np.uint8)
#         with multiprocessing.Pool(processes=nb_workers) as pool:
#             list(tqdm(pool.imap(_rasterize_chunk, args_list), total=len(args_list)))
#     return rasterized


# Define command-line functionality

def downsample_main(args):
    verbose_print(args, f'Downsampling {args.input} with factors {args.factor}')

    if args.tiff:
        os.makedirs(args.output, exist_ok=True)
        paths, filenames = utils.tifs_in_dir(args.input)
        for i, (path, filename) in enumerate(zip(paths, filenames)):
            verbose_print(args, f'Downsampling {filename}')
            arr = io.imread(path)
            if isinstance(args.factor, int):
                factors = tuple(args.factor for _ in range(arr.ndim))
            else:
                factors = tuple(args.factor)
            data = downsample(arr, factors)
            output = os.path.join(args.output, filename)
            io.imsave(output, data, compress=3)
    else:
        arr = io.open(args.input, mode='r')
        if isinstance(args.factor, int):
            factors = tuple(args.factor for _ in range(arr.ndim))
        else:
            factors = tuple(args.factor)
        data = downsample(arr, factors)
        verbose_print(args, f'Writing result to {args.output}')
        io.imsave(args.output, data, compress=3)

    verbose_print(args, f'Downsampling done!')


def downsample_cli(subparsers):
    downsample_parser = subparsers.add_parser('downsample', help="Downsample images",
                                              description='Image downsampling tool for segmentation')
    downsample_parser.add_argument('input', help="Path to input image to be downsampled")
    downsample_parser.add_argument('output', help="Path to output downsampled TIFF image")
    downsample_parser.add_argument('factor', help="Downsample factor", type=int, nargs='+')
    downsample_parser.add_argument('-t', '--tiff', help="TIFF folder flag", action='store_true')
    downsample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def ventricle_main(args):
    verbose_print(args, f'Segmenting ventricles in {args.input}')

    # Load the input image
    data = io.imread(args.input)

    # Load the model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = load_model(args.model, device)
    model = model.eval()
    verbose_print(args, f'Model successfully loaded from {args.model} to {device} device')

    # Segment the input image
    verbose_print(args, f'Segmentation progress:')
    output = segment_ventricles(model, data, args.t, device)

    # Remove border regions
    if args.exclude_border:
        verbose_print(args, f'Removing regions connected to image border')
        # This could also be done in 3D instead of slice-by-slice
        # I'm not sure if images will start in ventricle, so doing slice-by-slice to be safe
        img = np.zeros_like(output)
        for i, data in tqdm(enumerate(output), total=len(output)):
            img[i] = clear_border(data)
        output = img

    # Save the result to TIFF
    io.imsave(args.output, output, compress=3)
    verbose_print(args, f'Segmentation written to {args.output}')

    verbose_print(args, f'Ventricle segmentation done!')


def ventricle_cli(subparsers):
    ventricle_parser = subparsers.add_parser('ventricle', help="Segment ventricles",
                                             description='Ventricle segmentation tool using pretrained UNet')
    ventricle_parser.add_argument('input', help="Path to input (downsampled) image")
    ventricle_parser.add_argument('model', help="Path to pretrained Pytorch model")
    ventricle_parser.add_argument('output', help="Path to output ventricle segmentation TIFF")
    ventricle_parser.add_argument('-t', help="Probability threshold for binarization", type=float, default=0.5)
    ventricle_parser.add_argument('-e', '--exclude-border', help="Flag to exclude border regions", action='store_true')
    ventricle_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def foreground_main(args):
    verbose_print(args, f'Segmenting foreground from {args.input}')

    # Load the input image
    data = io.imread(args.input)

    # Smoothing
    if args.g is not None:
        data = gaussian_blur(data, args.g).astype(data.dtype)

    # Threshold image
    foreground = (data > args.t)  # .astype(np.uint8)

    # Fill holes
    # This is done slice-by-slice for now since there could be imaging problems where
    # a part of a ventricle is actually in the image at z = 0 or z = -1
    output = np.empty(foreground.shape, dtype=np.uint8)
    for i, img in enumerate(foreground):
        output[i] = binary_fill_holes(img)
    output *= 255

    # Save the result to TIFF
    io.imsave(args.output, output, compress=3)
    verbose_print(args, f'Segmentation written to {args.output}')

    verbose_print(args, f'Foreground segmentation done!')


def foreground_cli(subparsers):
    foreground_parser = subparsers.add_parser('foreground', help="Segment foreground",
                                              description='Foreground segmentation tool')
    foreground_parser.add_argument('input', help="Path to input (downsampled) image")
    foreground_parser.add_argument('output', help="Path to output foreground segmentation TIFF")
    foreground_parser.add_argument('-g', help="Amount of gaussian smoothing", type=float, nargs='+', default=None)
    foreground_parser.add_argument('-t', help="Intensity threshold", type=float, default=0.1)
    foreground_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def segment_main(args):
    commands_dict = {
        'downsample': downsample_main,
        'ventricle': ventricle_main,
        'foreground': foreground_main,
    }
    func = commands_dict.get(args.segment_command, None)
    if func is None:
        print("Pickle Rick uses segment subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'segment', '-h'])
    else:
        func(args)


def segment_cli(subparsers):
    segment_parser = subparsers.add_parser('segment', help="Segment organoid regions",
                                           description='Organoid region segmentation tool')
    segment_subparsers = segment_parser.add_subparsers(dest='segment_command', title='segment subcommands')
    downsample_cli(segment_subparsers)
    ventricle_cli(segment_subparsers)
    foreground_cli(segment_subparsers)
    return segment_parser
