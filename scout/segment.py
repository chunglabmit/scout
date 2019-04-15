"""
Segment Module
===============

This module performs organoid regions segmentation.

"""

import multiprocessing
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm
from scout import utils


# Segment ventricles with U-Net


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


def segment_cli(subparsers):
    segment_parser = subparsers.add_parser('segment', help="organoid region segmentation",
                                           description="Organoid region segmentation tool")
    segment_parser.add_argument('image', help="Path to nuclei image")
    segment_parser.add_argument('centroids', help="Path to nuclei centroids")
    segment_parser.add_argument('labels', help="Path to cell type labels")
    return segment_parser


def segment_main(args):
    print("Running segment main")
    print(args)


"""

SEGMENT
--------
segment-foreground
    zarr -> threshold -> smoothing -> foreground segmentation
segment-ventricles
    nuclei zarr -> ventricle probability -> ventricle segmentation
segment-regions
    niche labels -> rasterized segmentation -> smoothed segmentation
combine-segmentations
    foreground seg + ventricle seg + region seg -> combined segmentation
    
"""
