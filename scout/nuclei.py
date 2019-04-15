"""
Nuclei Module
==============

This module performs nuclei detection, segmentation, and cytometry.

These include the following subcommands:
    - detect : detect all nuclei in image
    - segment : segment all detected nuclei
    - fluorescence : measure fluorescence for each cell
    - gate : assign cell-type labels by fluorescence thresholding
    - morphology : compute morphological features of segmented nuclei

"""

import subprocess
import warnings
from functools import partial
import multiprocessing
import tempfile
import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import morphology
from skimage import measure
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import matplotlib.pyplot as plt
from scout import io
from scout.preprocess import gaussian_blur_parallel
from scout import detection
from scout import utils
from scout.utils import verbose_print
from scout.synthetic import points_to_binary


# Nuclei segmentation


def watershed_centers(image, centers, mask, **watershed_kwargs):
    seeds = points_to_binary(tuple(centers.T), image.shape, cval=1)
    markers = ndi.label(seeds)[0]
    labels = morphology.watershed(-image, markers, mask=mask, **watershed_kwargs)
    return labels


def _watershed_probability_chunk(input_tuple, output, centers, mask, overlap, **watershed_kwargs):
    arr, start_coord, chunks = input_tuple

    # extract ghosted chunk of data
    data_overlap, start_ghosted, stop_ghosted = utils.extract_ghosted_chunk(arr, start_coord, chunks, overlap)
    mask_overlap, _, _ = utils.extract_ghosted_chunk(mask, start_coord, chunks, overlap)

    # Find seeds within the ghosted chunk
    centers_internal = utils.filter_points_in_box(centers, start_ghosted, stop_ghosted)
    centers_internal_local = centers_internal - start_ghosted

    # segment the chunk
    labels_overlap = watershed_centers(data_overlap,
                                       centers_internal_local,
                                       mask_overlap,
                                       watershed_line=True)
    binary_overlap = (labels_overlap > 0)
    # binary_overlap_eroded = ndi.binary_erosion(binary_overlap)

    # write the segmentation result
    start_local = start_coord - start_ghosted
    stop_local = np.minimum(start_local + np.asarray(chunks), np.asarray(arr.shape) - start_ghosted)
    # binary_seg = utils.extract_box(binary_overlap_eroded, start_local, stop_local)
    binary_seg = utils.extract_box(binary_overlap, start_local, stop_local)
    stop_coord = start_coord + np.asarray(binary_seg.shape)
    utils.insert_box(output, start_coord, stop_coord, binary_seg)


def watershed_centers_parallel(prob, centers, mask, output, chunks, overlap, nb_workers=None):
    f = partial(_watershed_probability_chunk,
                output=output,
                centers=centers,
                mask=mask,
                overlap=overlap)
    utils.pmap_chunks(f, prob, chunks, nb_workers, use_imap=True)


# Fluorescence sampling, statistics, and gating


def sample_intensity_cube(center, image, radius):
    start = [max(0, int(c - radius)) for c in center]
    stop = [min(int(c + radius + 1), d - 1) for c, d in zip(center, image.shape)]
    bbox = utils.extract_box(image, start, stop)
    return bbox.flatten()


def nuclei_centered_intensities(image, centers, radius, mode='cube', nb_workers=None):
    if nb_workers is None:
        nb_workers = multiprocessing.cpu_count()

    if mode == 'cube':
        f = partial(sample_intensity_cube, image=image, radius=radius)
    else:
        raise ValueError('Only cube sampling is currently supported')

    with multiprocessing.Pool(nb_workers) as pool:
        intensities = list(tqdm(pool.imap(f, centers), total=centers.shape[0]))

    return intensities


def calculate_mfi(input):
    """Calculate the Mean Fluorescence Intensity (MFI) for input list of nucleus-centered samples

    Parameters
    ----------
    input : list
        List of ndarrays containing image intensities near nuclei

    Returns
    -------
    output : ndarray
        1D array of MFIs for each nucleus
    """
    return np.asarray([x.mean() for x in input])


def calculate_stdev(input):
    """Calculate the standard deviation for input list of nucleus-centered samples

    Parameters
    ----------
    input : list
        List of ndarrays containing image intensities near nuclei

    Returns
    -------
    output : ndarray
        1D array of standard deviations for each nucleus
    """
    return np.asarray([x.std() for x in input])


def threshold_mfi(mfi, threshold):
    positive_idx = np.where(mfi > threshold)[0]
    labels = np.zeros(mfi.shape, dtype=np.int)
    labels[positive_idx] = 1
    return labels


# Nuclei morphological features


def morphological_features(seg):
    props = measure.regionprops(seg)
    nb_labels = len(props)
    centers = np.zeros((nb_labels, seg.ndim))
    volumes = np.zeros(nb_labels)
    eq_diams = np.zeros(nb_labels)
    minor_lengths = np.zeros(nb_labels)
    major_lengths = np.zeros(nb_labels)
    for i, region in tqdm(enumerate(props), total=nb_labels):
        centers[i] = region.centroid
        volumes[i] = region.area
        eq_diams[i] = region.equivalent_diameter
        minor_lengths[i] = region.minor_axis_length
        major_lengths[i] = region.major_axis_length
    axis_ratios = major_lengths / np.clip(minor_lengths, 1, None)
    return centers, volumes, eq_diams, minor_lengths, major_lengths, axis_ratios


# Define command-line main functions


def detect_main(args):
    # Open nuclei Zarr array
    verbose_print(args, f'Detecting nuclei in {args.input}')
    arr = io.open(args.input, mode='r')
    shape, dtype, chunks = arr.shape, arr.dtype, arr.chunks
    verbose_print(args, f'Opened image: {shape} {dtype}')

    # Create probability Zarr array
    prob_arr = io.new_zarr(args.probability, shape=shape, chunks=chunks, dtype='float32')

    # Detect nuclei
    centroids = detection.detect_nuclei_parallel(arr,
                                                 sigma=args.g,
                                                 min_intensity=args.m,
                                                 steepness=args.s,
                                                 offset=args.b,
                                                 I0=args.r,
                                                 stdev=args.x,
                                                 prob_thresh=args.p,
                                                 min_dist=args.d,
                                                 chunks=tuple(args.c),
                                                 overlap=args.o,
                                                 nb_workers=1,  # GPU is one worker
                                                 prob_output=prob_arr)
    nb_centroids = centroids.shape[0]
    verbose_print(args, f'Found {nb_centroids} nuclei centroids')

    # Save centroids
    np.save(args.output, centroids)
    verbose_print(args, f'Saved centroids to {args.output}')

    verbose_print(args, f'Nuclei detection done!')


def segment_main(args):
    # Open probability map Zarr array
    verbose_print(args, f'Segmenting nuclei in {args.input}')
    prob_arr = io.open(args.input, mode='r')
    shape, dtype, chunks = prob_arr.shape, prob_arr.dtype, prob_arr.chunks
    verbose_print(args, f'Opened image: {shape} {dtype}')
    if dtype != 'float32':
        warnings.warn('Input dtype is not float32... may not have passed a probability map')

    # Load nuclei centroids
    centroids = np.load(args.centroids)
    nb_centroids = centroids.shape[0]

    # TODO: Move the foreground and segmentation Zarr arrays to temporary directory by default

    # Create watershed mask by thresholding the probability map
    prob = prob_arr[:]
    foreground = (prob > args.t).astype(np.uint8)
    foreground_arr = io.new_zarr(args.foreground, shape=shape, chunks=chunks, dtype='uint8')
    foreground_arr[:] = foreground

    # Create output Zarr array for binary segmentation
    binary_seg = io.new_zarr(args.binary, shape, chunks, 'uint8')

    # segment and label nuclei
    verbose_print(args, f'Performing watershed with {args.w} workers')
    watershed_centers_parallel(prob_arr,
                               centers=centroids,
                               mask=foreground_arr,
                               output=binary_seg,
                               chunks=chunks,
                               overlap=args.o,
                               nb_workers=args.w)
    labeled, nb_lbls = ndi.label(binary_seg[:])
    verbose_print(args, f'{nb_lbls} labeled nuclei in segmentation ({100*nb_lbls/nb_centroids:.1f}% of detected)')

    # Save the labeled segmentation
    io.imsave(args.output, labeled, compress=3)

    verbose_print(args, 'Nuclei segmentation done!')


def fluorescence_main(args):
    nb_images = len(args.input)
    verbose_print(args, f'Passed {nb_images} images to measure fluorescence')

    # Load centroids
    centroids = np.load(args.centroids)

    # Initialize output arrays
    mfis = np.zeros((centroids.shape[0], nb_images))
    stdevs = np.zeros((centroids.shape[0], nb_images))

    for i, path in enumerate(args.input):
        # Open image
        arr = io.open(path, mode='r')
        shape, dtype, chunks = arr.shape, arr.dtype, arr.chunks
        verbose_print(args, f'Sampling from {path}: {shape} {dtype}')

        # Sample image
        if args.g is not None:
            # Perform smoothing in a temporary array
            verbose_print(args, f'Smoothing {path} with sigma {tuple(args.g)}')
            with tempfile.TemporaryDirectory() as temp_path:
                smoothed_arr = io.new_zarr(temp_path, shape, chunks, dtype)
                gaussian_blur_parallel(arr, args.g, smoothed_arr, arr.chunks, args.o, args.w)
                intensities = nuclei_centered_intensities(smoothed_arr, centroids, args.r, mode=args.m, nb_workers=args.w)
            # Temporary array deleted when context ends
        else:
            intensities = nuclei_centered_intensities(arr, centroids, args.r, mode=args.m, nb_workers=args.w)

        # Compute statistics
        mfis[:, i] = calculate_mfi(intensities)
        stdevs[:, i] = calculate_stdev(intensities)

    # Save stats
    np.save(args.mfi, mfis)
    np.save(args.stdev, stdevs)

    verbose_print(args, f'Fluorescence measurements done!')


def gate_main(args):
    verbose_print(args, f'Gating cells based on fluorescence in {args.input}')

    # Load MFIs and check for mismatch
    mfis = np.load(args.input)
    if mfis.shape[-1] != len(args.thresholds):
        raise ValueError('Number of thresholds must match the number of channels in MFI array')

    # Show plot
    if args.plot:
        verbose_print(args, f'Showing cytometry plot...')

        mfi_x, mfi_y = mfis[:, args.x], mfis[:, args.y]

        plt.hist2d(mfi_x, mfi_y, bins=args.b)
        plt.plot([args.thresholds[0], args.thresholds[0]], [0, mfi_y.max()], 'r-')
        plt.plot([0, mfi_x.max()], [args.thresholds[1], args.thresholds[1]], 'r-')
        plt.xlim([0, mfi_x.max()])
        plt.ylim([0, mfi_y.max()])
        plt.xlabel(f'MFI column {args.x}')
        plt.ylabel(f'MFI column {args.y}')
        plt.show()

    # Gate each channel
    labels = np.asarray([threshold_mfi(mfi, t) for mfi, t in zip(mfis.T, args.thresholds)], dtype=np.uint8).T

    # Save the result
    np.save(args.output, labels)
    verbose_print(args, f'Gating results written to {args.output}')

    verbose_print(args, f'Gating cells done!')


def morphology_main(args):
    verbose_print(args, f'Computing morphological features for {args.input}')

    # Load nuclei segmentation
    seg = io.imread(args.input).astype(np.int32)

    # Compute morphological features
    centers, volumes, eq_diams, minor_lengths, major_lengths, axis_ratios = morphological_features(seg)

    # Load the detected centroids
    centroids = np.load(args.centroids)
    verbose_print(args, f'Cross-referencing centroids in {args.centroids}')

    # Match segmentation to nearest centroid
    nbrs = NearestNeighbors(n_neighbors=1).fit(centers)
    distances, indices = nbrs.kneighbors(centroids)
    centroids_idx = indices[:, 0].astype(np.int)  # Index into segmented centers for each detected centroid

    # Save CSV containing morphologies for each detected centroid
    data = {
        'z': centroids[:, 0],
        'y': centroids[:, 1],
        'x': centroids[:, 2],
        'com_z': centers[:, 0][centroids_idx],
        'com_y': centers[:, 1][centroids_idx],
        'com_x': centers[:, 2][centroids_idx],
        'com_idx': centroids_idx,
        'volume': volumes[centroids_idx],
        'eq_diam': eq_diams[centroids_idx],
        'minor_length': minor_lengths[centroids_idx],
        'major_length': major_lengths[centroids_idx],
        'axis_ratio': axis_ratios[centroids_idx],
    }
    df = pd.DataFrame(data)
    df.to_csv(args.output)
    verbose_print(args, f'Morphological features written to {args.output}')

    verbose_print(args, f'Computing morphologies done!')


# Define command-line interfaces


def detect_cli(subparsers):
    detect_parser = subparsers.add_parser('detect', help="Detect all nuclei centroids in image",
                                          description='Detects nuclei centroids using a curvature-based filter')
    detect_parser.add_argument('input', help="Path to nuclei image Zarr array")
    detect_parser.add_argument('probability', help="Path to nuclei probability map Zarr array")
    detect_parser.add_argument('output', help="Path to save numpy array of nuclei centroids")
    detect_parser.add_argument('-g', help="Amount of gaussian blur", type=float, nargs='+', default=(1.2, 2.0, 2.0))
    detect_parser.add_argument('-s', help="Steepness of curvature filter", type=float, default=4000)
    detect_parser.add_argument('-b', help="Bias of curvature filter", type=float, default=-0.0001)
    detect_parser.add_argument('-r', help="Reference intensity prior", type=float, default=500)
    detect_parser.add_argument('-x', help="Crossover intensity prior", type=float, default=1e-5)
    detect_parser.add_argument('-d', help="Minimum distance between centroids", type=float, default=2)
    detect_parser.add_argument('-p', help="Minimum probability of a centroid", type=float, default=0.2)
    detect_parser.add_argument('-c', help="Chunk shape to process at a time", type=int, nargs='+', default=3 * (64,))
    detect_parser.add_argument('-m', help="Minimum intensity to skip chunk", type=float, default=500)
    detect_parser.add_argument('-o', help="Overlap in pixels between chunks", type=int, default=4)
    detect_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def segment_cli(subparsers):
    segment_parser = subparsers.add_parser('segment', help="Segment all nuclei from probability map",
                                           description='Segments all nuclei using 3D watershed')
    segment_parser.add_argument('input', help="Path to nuclei probability map Zarr array")
    segment_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    segment_parser.add_argument('foreground', help="Path to nuclei foreground Zarr array")
    segment_parser.add_argument('binary', help="Path to nuclei binary segmentation Zarr array")
    segment_parser.add_argument('output', help="Path to labeled nuclei TIFF image")
    segment_parser.add_argument('-t', help="Probability threshold for segmentation", type=float, default=0.1)
    segment_parser.add_argument('-w', help="Number of workers for segmentation", type=int, default=None)
    segment_parser.add_argument('-o', help="Overlap in pixels between chunks", type=int, default=4)
    segment_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def fluorescence_cli(subparsers):
    fluorescence_parser = subparsers.add_parser('fluorescence', help="Measure fluorescence for each cell",
                                                description='Measures fluorescence statistics at each centroid')
    fluorescence_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    fluorescence_parser.add_argument('mfi', help="Path to output MFI numpy array")
    fluorescence_parser.add_argument('stdev', help="Path to output StDev numpy array")
    fluorescence_parser.add_argument('input', help="Path to input images to sample from", nargs='+')
    fluorescence_parser.add_argument('-g', help="Amount of gaussian blur", type=float, nargs='+', default=None)
    fluorescence_parser.add_argument('-m', help="Sampling mode {'cube'}", type=str, default='cube')
    fluorescence_parser.add_argument('-r', help="Sampling radius", type=int, default=1)
    fluorescence_parser.add_argument('-w', help="Number of workers for segmentation", type=int, default=None)
    fluorescence_parser.add_argument('-o', help="Overlap in pixels between chunks for smoothing", type=int, default=4)
    fluorescence_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def gate_cli(subparsers):
    gate_parser = subparsers.add_parser('gate', help="Gate cells based on fluorescence",
                                        description='Sets gates and classifies cell-types based on fluorescence')
    gate_parser.add_argument('input', help="Path to input MFI numpy array")
    gate_parser.add_argument('output', help="Path to output labels numpy array")
    gate_parser.add_argument('thresholds', help="MFI gates for each channel", nargs="+", type=float)
    gate_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    gate_parser.add_argument('-b', help="Number of bins to use in historgram", type=int, default=128)
    gate_parser.add_argument('-x', help="MFI column index for x-axis", type=int, default=0)
    gate_parser.add_argument('-y', help="MFI column index for y-axis", type=int, default=1)
    gate_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def morphology_cli(subparsers):
    morphology_parser = subparsers.add_parser('morphology', help="Measure morphological features of nuclei",
                                              description='Uses nuclei segmentation to compute morphological features')
    morphology_parser.add_argument('input', help="Path to input nuclei segmentation TIFF")
    morphology_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    morphology_parser.add_argument('output', help="Path to output morphological features CSV")
    morphology_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def nuclei_cli(subparsers):
    nuclei_parser = subparsers.add_parser('nuclei', help="nuclei detection and segmentation",
                                          description='Nuclei detection and segmentation tool')
    nuclei_subparsers = nuclei_parser.add_subparsers(dest='nuclei_command', title='nuclei subcommands')
    detect_cli(nuclei_subparsers)
    segment_cli(nuclei_subparsers)
    fluorescence_cli(nuclei_subparsers)
    gate_cli(nuclei_subparsers)
    morphology_cli(nuclei_subparsers)
    return nuclei_parser


# Main nuclei entry point


def nuclei_main(args):
    commands_dict = {
        'detect': detect_main,
        'segment': segment_main,
        'fluorescence': fluorescence_main,
        'gate': gate_main,
        'morphology': morphology_main,
    }
    func = commands_dict.get(args.nuclei_command, None)
    if func is None:
        print("Pickle Rick uses nuclei subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'nuclei', '-h'])
    else:
        func(args)
