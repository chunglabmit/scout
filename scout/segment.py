"""
Segment Module
===============

This module performs organoid regions segmentation.

"""

import subprocess
import multiprocessing
import numpy as np
from tqdm import tqdm
from scipy.interpolate import griddata
from skimage.transform import downscale_local_mean
import torch
from scout import io
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
    # Load image
    data = arr[:]
    # Resize image
    output = downscale_local_mean(data, factors).astype(arr.dtype)
    return output


# Segment ventricles with U-Net

def load_model(path, device):
    model = UNet(1, 1)
    model.load_state_dict(torch.load(path))
    model.to(device)
    return model


# Calculate local densities and threshold


# rasterize region labels


rasterized = None


def _rasterize_chunk(args):
    start, shape, chunks, pts, labels = args
    global rasterized
    stop = np.minimum(shape, start + np.asarray(chunks))
    grid_z, grid_y, grid_x = np.mgrid[start[0]:stop[0], start[1]:stop[1], start[2]:stop[2]]
    data = griddata(pts, labels, (grid_z, grid_y, grid_x), method='nearest').astype(np.uint8)
    rasterized = utils.insert_box(rasterized, start, stop, data)


def rasterize_regions(pts, labels, shape, chunks=None, nb_workers=None):
    global rasterized
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()
    if chunks is None:
        grid_z, grid_y, grid_x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]]
        rasterized = griddata(pts, labels, (grid_z, grid_y, grid_x), method='nearest').astype(np.uint8)
    else:
        chunk_coords = utils.chunk_coordinates(shape, chunks)
        args_list = []
        for start in tqdm(chunk_coords, total=len(chunk_coords)):
            args_list.append((start, shape, chunks, pts, labels))
        rasterized = utils.SharedMemory(shape=shape, dtype=np.uint8)
        with multiprocessing.Pool(processes=nb_workers) as pool:
            list(tqdm(pool.imap(_rasterize_chunk, args_list), total=len(args_list)))
    return rasterized

# Smooth with graph-cuts


# Define command-line functionality

def downsample_main(args):
    if isinstance(args.factor, int):
        factors = tuple(args.factor for _ in range(arr.ndim))
    else:
        factors = tuple(args.factor)

    verbose_print(args, f'Downsampling {args.input} with factors {factors}')

    arr = io.open(args.input, mode='r')

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
    downsample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def ventricle_main(args):
    verbose_print(args, f'Segmenting ventricles in {args.input}')

    data = io.imread(args.input)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, device)

    verbose_print(args, f'Model successfully loaded from {args.model} to {device} device')

    for img in tqdm(data, total=len(data)):
        img_tensor = torch.from_numpy(img.astype(np.float32)).to(device)
        prob = model(img_tensor)
        print(prob.max())



def ventricle_cli(subparsers):
    ventricle_parser = subparsers.add_parser('ventricle', help="Downsample images",
                                              description='Image downsampling tool for segmentation')
    ventricle_parser.add_argument('input', help="Path to input (downsampled) image")
    ventricle_parser.add_argument('model', help="Path to pretrained Pytorch model")
    ventricle_parser.add_argument('output', help="Path to output ventricle segmentation TIFF")
    ventricle_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')



def segment_main(args):
    commands_dict = {
        'downsample': downsample_main,
        'ventricle': ventricle_main,
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
    return segment_parser


"""

SEGMENT
--------
segment-foreground
    zarr image -> threshold -> smoothing -> foreground segmentation
segment-ventricles
    nuclei zarr -> ventricle UNet probability -> ventricle segmentation
segment-regions
    niche labels -> rasterized segmentation -> smoothed segmentation
combine-segmentations
    foreground seg + ventricle seg + region seg -> combined segmentation
    
"""
