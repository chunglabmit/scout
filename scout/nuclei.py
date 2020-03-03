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
    - name : assign names to each cell-type

"""

import subprocess
import os
import warnings
from functools import partial
import multiprocessing
import tempfile
import numpy as np
import pandas as pd
import zarr
from scipy import ndimage as ndi
from skimage import morphology
from skimage.measure import regionprops
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
from scout import io
from scout.preprocess import gaussian_blur_parallel
from scout import detection
from scout import utils
from scout.utils import verbose_print
from scout.synthetic import points_to_binary
from scout.niche import name_cli, name_main
import matplotlib
try:
    #matplotlib.use('tkagg')
    matplotlib.use('Agg')
except:
    pass
import matplotlib.pyplot as plt
from matplotlib import colors


# Nuclei segmentation

def _threshold_chunk(inputs, threshold, output):
    arr, start, chunks = inputs
    prob, _, stop = utils.extract_ghosted_chunk(arr, start, chunks, overlap=0)
    foreground = (prob > threshold).astype(np.uint8)
    utils.insert_box(output, start, stop, foreground)


def watershed_centers(image, centers, mask, **watershed_kwargs):
    seeds = points_to_binary(tuple(centers.T), image.shape, cval=1)
    markers = ndi.label(seeds)[0]
    labels = morphology.watershed(-image, markers, mask=mask, **watershed_kwargs)
    return labels


def _watershed_probability_chunk(input_tuple, output, centers, mask, overlap, **watershed_kwargs):
    arr, start_coord, chunks = input_tuple

    # extract ghosted chunk of data
    mask_overlap, start_ghosted, stop_ghosted = utils.extract_ghosted_chunk(mask, start_coord, chunks, overlap)
    if not np.any(mask_overlap):
        # # write zeros for blank chunk
        # start_local = start_coord - start_ghosted
        # stop_local = np.minimum(start_local + np.asarray(chunks), np.asarray(arr.shape) - start_ghosted)
        # binary_seg = np.zeros(tuple(stop_local - start_local), output.dtype)
        # stop_coord = start_coord + np.asarray(binary_seg.shape)
        # utils.insert_box(output, start_coord, stop_coord, binary_seg)
        return
    data_overlap, _, _ = utils.extract_ghosted_chunk(arr, start_coord, chunks, overlap)

    # Find seeds within the ghosted chunk
    centers_internal = utils.filter_points_in_box(centers, start_ghosted, stop_ghosted)
    centers_internal_local = centers_internal - start_ghosted
    # Skip if no centers in ghosted chunk
    if len(centers_internal) == 0:
        return

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
    utils.pmap_chunks(f, prob, chunks, nb_workers, use_imap=True, unordered=True, chunksize=10)
    # utils.pmap_chunks(f, prob, chunks, 2, use_imap=True)


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

def segment_centroid(centroid, window_size, binary_seg, return_seg=False):
    window_size = np.asarray(window_size)

    # Extract ROI centered on centroid
    start = np.maximum(np.zeros(3), centroid - window_size // 2).astype(np.int)
    stop = np.minimum(np.asarray(binary_seg.shape), centroid + window_size // 2).astype(np.int)
    patch = utils.extract_box(binary_seg, start, stop)
    lbl, _ = ndi.label(patch)

    # Extract pixels with the same label as the centroid
    centroid_local = centroid - start
    value = lbl[centroid_local[0], centroid_local[1], centroid_local[2]]
    if value > 0:
        # print('Found segmentation!')
        single = (lbl == value).astype(np.uint8)
    else:
        print('No foreground on centroid point!')
        single = np.zeros(patch.shape, np.uint8)
        single[centroid_local[0], centroid_local[1], centroid_local[2]] = 1

    # Compute morphological features
    region = regionprops(single)[0]
    center = region.centroid + start
    volume = region.area
    eq_diam = region.equivalent_diameter
    minor_length = region.minor_axis_length
    major_length = region.major_axis_length
    axis_ratio = major_length / np.clip(minor_length, 1, None)
    features = np.array([center[0], center[1], center[2], volume, eq_diam, minor_length, major_length, axis_ratio])

    # Pad to window_size if needed
    # This needs to happen after center of mass is computed because padding will offset from original start
    if single.shape != tuple(window_size):
        middle = window_size // 2
        start_offset = tuple(np.clip(middle - centroid_local, 0, None))  # pre-padding
        stop_offset = tuple(np.clip(middle - (stop - centroid), 0, None))  # post-padding
        pad_width = tuple(zip(start_offset, stop_offset))
        single = np.pad(single, pad_width, 'constant')

    if return_seg:
        return features, single
    else:
        return (features,)


def _segment_centroid(inputs):
    return segment_centroid(*inputs)


def morphological_features(seg):
    # Old code, replaced with centroid labeling
    props = regionprops(seg)
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
    if args.voxel_size is not None and args.output_um is None:
        raise ValueError('A path to output_um array must be specified if given voxel dimensions')
    elif args.voxel_size is None and args.output_um is not None:
        raise ValueError('Voxel size must be specified if path to output_um is given')

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

    # Convert to micron if possible
    if args.voxel_size is not None:
        voxel_size = utils.read_voxel_size(args.voxel_size)
        centroids_um = centroids * np.asarray(voxel_size)

    # Save centroids
    np.save(args.output, centroids)
    verbose_print(args, f'Saved centroids to {args.output}')
    if args.output_um is not None:
        np.save(args.output_um, centroids_um)
        verbose_print(args, f'Saved centroids in micron to {args.output_um}')

    verbose_print(args, f'Nuclei detection done!')


def detect_cli(subparsers):
    detect_parser = subparsers.add_parser('detect', help="Detect all nuclei centroids in image",
                                          description='Detects nuclei centroids using a curvature-based filter')
    detect_parser.add_argument('input', help="Path to nuclei image Zarr array")
    detect_parser.add_argument('probability', help="Path to nuclei probability map Zarr array")
    detect_parser.add_argument('output', help="Path to save numpy array of nuclei centroids")
    detect_parser.add_argument('--voxel-size', help="Path to voxel size CSV", default=None)
    detect_parser.add_argument('--output-um', help="Path to save numpy array of centroids in micron", default=None)
    detect_parser.add_argument('-g', help="Amount of gaussian blur", type=float, nargs='+', default=(1.0, 3.0, 3.0))
    detect_parser.add_argument('-s', help="Steepness of curvature filter", type=float, default=600)
    detect_parser.add_argument('-b', help="Bias of curvature filter", type=float, default=-0.0005)
    detect_parser.add_argument('-r', help="Reference intensity prior", type=float, default=1.0)
    detect_parser.add_argument('-x', help="Crossover intensity prior", type=float, default=0.10)
    detect_parser.add_argument('-d', help="Minimum distance between centroids", type=float, default=3)
    detect_parser.add_argument('-p', help="Minimum probability of a centroid", type=float, default=0.2)
    detect_parser.add_argument('-c', help="Chunk shape to process at a time", type=int, nargs='+', default=3 * (64,))
    detect_parser.add_argument('-m', help="Minimum intensity to skip chunk", type=float, default=0.1)
    detect_parser.add_argument('-o', help="Overlap in pixels between chunks", type=int, default=8)
    detect_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def segment_main(args):
    if args.n is None:
        nb_workers = multiprocessing.cpu_count()
    else:
        nb_workers = args.n

    # Open probability map Zarr array
    verbose_print(args, f'Segmenting nuclei in {args.input}')
    prob_arr = io.open(args.input, mode='r')
    shape, dtype, chunks = prob_arr.shape, prob_arr.dtype, prob_arr.chunks
    verbose_print(args, f'Opened image: {shape} {dtype}')
    if dtype != 'float32':
        warnings.warn('Input dtype is not float32... may not have passed a probability map')

    # Load nuclei centroids
    centroids = np.load(args.centroids)

    # Create foreground mask by thresholding the probability map
    verbose_print(args, f'Thresholding probability at {args.t}, writing foreground to {args.foreground}')
    foreground_arr = io.new_zarr(args.foreground, shape=shape, chunks=chunks, dtype='uint8')
    f = partial(_threshold_chunk, threshold=args.t, output=foreground_arr)
    utils.pmap_chunks(f, prob_arr, chunks, 1, use_imap=True)

    # Add watershed lines to the foreground mask to break up touching nuclei
    verbose_print(args, f'Performing watershed, writing binary segmentation to {args.output}')
    binary_seg = io.new_zarr(args.output, shape, chunks, 'uint8')
    watershed_centers_parallel(prob_arr,
                               centers=centroids,
                               mask=foreground_arr,
                               output=binary_seg,
                               chunks=chunks,
                               overlap=args.o,
                               nb_workers=nb_workers)

    verbose_print(args, 'Nuclei segmentation done!')


def segment_cli(subparsers):
    segment_parser = subparsers.add_parser('segment', help="Segment all nuclei from probability map",
                                           description='Segments all nuclei to binary using 3D watershed')
    segment_parser.add_argument('input', help="Path to nuclei probability map Zarr array")
    segment_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    segment_parser.add_argument('foreground', help="Path to nuclei foreground Zarr array")
    segment_parser.add_argument('output', help="Path to nuclei binary segmentation Zarr array")
    segment_parser.add_argument('-t', help="Probability threshold for segmentation", type=float, default=0.1)
    segment_parser.add_argument('-n', help="Number of workers for segmentation", type=int, default=None)
    segment_parser.add_argument('-o', help="Overlap in pixels between chunks", type=int, default=8)
    segment_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def fluorescence_main(args):
    if isinstance(args.inputs, list):
        inputs = args.inputs
    else:
        inputs = [args.inputs]

    nb_images = len(inputs)
    verbose_print(args, f'Passed {nb_images} images to measure fluorescence')

    # Load centroids
    centroids = np.load(args.centroids)

    # Initialize output arrays
    mfis = np.zeros((centroids.shape[0], nb_images))
    stdevs = np.zeros((centroids.shape[0], nb_images))
    for i, path in enumerate(inputs):
        # Open image
        arr = io.open(path, mode='r')
        shape, dtype, chunks = arr.shape, arr.dtype, arr.chunks
        verbose_print(args, f'Sampling from {path}: {shape} {dtype}')

        # Sample image
        if args.g is not None:
            # Perform smoothing in a temporary array
            verbose_print(args, f'Smoothing {path} with sigma {tuple(args.g)}')
            with tempfile.TemporaryDirectory(prefix=os.path.abspath('.')) as temp_path:
                smoothed_arr = io.new_zarr(temp_path, shape, chunks, dtype)
                gaussian_blur_parallel(arr, args.g, smoothed_arr, arr.chunks, args.o, args.w)  # Too many workers gives Zarr race condition
                verbose_print(args, f'Sampling fluorescence from smoothed {path}')
                intensities = nuclei_centered_intensities(smoothed_arr, centroids, args.r, mode=args.m, nb_workers=args.w)
            # Temporary array deleted when context ends
        else:
            intensities = nuclei_centered_intensities(arr, centroids, args.r, mode=args.m, nb_workers=args.w)

        # Compute statistics
        mfis[:, i] = calculate_mfi(intensities)
        stdevs[:, i] = calculate_stdev(intensities)

    # Make output folder
    os.makedirs(args.output, exist_ok=True)

    # Save numpy array of MFIs and stdevs
    mfi_path = os.path.join(args.output, 'nuclei_mfis.npy')
    np.save(mfi_path, mfis)
    verbose_print(args, f'MFIs written to {mfi_path}')

    stdev_path = os.path.join(args.output, 'nuclei_stdevs.npy')
    np.save(stdev_path, stdevs)
    verbose_print(args, f'StDevs written to {stdev_path}')

    # Save CSV containing morphologies for each detected centroid
    # sox2.zarr/ <-- forward slash makes os.path.basename eval to empty string
    # Can use os.path.dirname(path) to get sox2.zarr, then use basename on that
    basenames = [os.path.basename(os.path.dirname(path)).split('.')[0] for path in inputs]
    csv_names = ['fluorescence_' + str(base) + '.csv' for base in basenames]
    csv_paths = [os.path.join(args.output, name) for name in csv_names]
    for i, (base, path) in enumerate(zip(basenames, csv_paths)):
        df = pd.DataFrame({'mfi': mfis[:, i], 'stdev': stdevs[:, i]})
        df.to_csv(path)
        verbose_print(args, f'Fluorescence statistics for {base} written to {path}')

    verbose_print(args, f'Fluorescence measurements done!')


def fluorescence_cli(subparsers):
    fluorescence_parser = subparsers.add_parser('fluorescence', help="Measure fluorescence for each cell",
                                                description='Measures fluorescence statistics at each centroid')
    fluorescence_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    fluorescence_parser.add_argument('output', help="Path to output folder to save fluorescence CSVs")
    fluorescence_parser.add_argument('inputs', help="Path to input images to sample from", nargs='+')
    fluorescence_parser.add_argument('-g', help="Amount of gaussian blur", type=float, nargs='+', default=None)
    fluorescence_parser.add_argument('-m', help="Sampling mode {'cube'}", type=str, default='cube')
    fluorescence_parser.add_argument('-r', help="Sampling radius", type=int, default=1)
    fluorescence_parser.add_argument('-w', help="Number of workers", type=int, default=None)
    fluorescence_parser.add_argument('-o', help="Overlap in pixels between chunks for smoothing", type=int, default=8)
    fluorescence_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')

# scout niche radial tests/data/centroids_um.npy tests/data/gate_lp="Verbose flag", action='store_true')


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

        if args.r is None:
            x_max = mfi_x.max()
            y_max = mfi_y.max()
        else:
            x_max = args.r[0]
            y_max = args.r[1]

        plt.hist2d(mfi_x, mfi_y, bins=args.b, norm=colors.PowerNorm(0.25), range=((0, x_max), (0, y_max)))
        plt.plot([args.thresholds[0], args.thresholds[0]], [0, y_max], 'r-')
        plt.plot([0, x_max], [args.thresholds[1], args.thresholds[1]], 'r-')
        plt.xlim([0, x_max])
        plt.ylim([0, y_max])
        plt.xlabel(f'MFI column {args.x}')
        plt.ylabel(f'MFI column {args.y}')

        fig_output = os.path.join(os.path.dirname(args.output),"gating_channels.png")
        plt.savefig(fig_output, dpi=600)
        #plt.show()

    # Gate each channel
    labels = np.asarray([threshold_mfi(mfi, t) for mfi, t in zip(mfis.T, args.thresholds)], dtype=np.uint8).T
    # TODO: Add DN labels in here

    # Save the result
    np.save(args.output, labels)
    verbose_print(args, f'Gating results written to {args.output}')

    verbose_print(args, f'Gating cells done!')


def gate_cli(subparsers):
    gate_parser = subparsers.add_parser('gate', help="Gate cells based on fluorescence",
                                        description='Gates cells and classify cell-types based on fluorescence')
    gate_parser.add_argument('input', help="Path to input MFI numpy array")
    gate_parser.add_argument('output', help="Path to output labels numpy array")
    gate_parser.add_argument('thresholds', help="MFI gates for each channel", nargs="+", type=float)
    gate_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    gate_parser.add_argument('-b', help="Number of bins to use in historgram", type=int, default=128)
    gate_parser.add_argument('-r', help="Ranges for each axis", nargs="+", type=float, default=None)
    gate_parser.add_argument('-x', help="MFI column index for x-axis", type=int, default=0)
    gate_parser.add_argument('-y', help="MFI column index for y-axis", type=int, default=1)
    gate_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def morphology_main(args):
    if args.n is None:
        nb_workers = multiprocessing.cpu_count()
    else:
        nb_workers = args.n

    if args.segmentations is not None:
        return_seg = True
    else:
        return_seg = False

    verbose_print(args, f'Computing morphological features for {args.input}')

    # Get window size
    window_size = np.asarray(args.w)
    verbose_print(args, f'Using window size of {window_size} around each cell')

    # Load the detected centroids and open binary segmentation
    centroids = np.load(args.centroids)  # TODO: Make this consider voxel dimensions
    binary_seg = io.open(args.input, mode='r')

    # Compute labeled segmentation and morphologies for each cell
    if return_seg:
        verbose_print(args, f'Computing segmentations and morphologies with {nb_workers} workers')
    else:
        verbose_print(args, f'Computing morphologies with {nb_workers} workers')
    args_list = [(centroid, window_size, binary_seg, return_seg) for centroid in centroids]
    with multiprocessing.Pool(nb_workers) as pool:
        results = list(tqdm(pool.imap(_segment_centroid, args_list), total=len(args_list)))
    # Unpack morphological features
    # features = np.array([center, volume, eq_diam, minor_length, major_length, axis_ratio])
    features = np.asarray([r[0] for r in results])  # N x feats
    centers_z = features[:, 0]
    centers_y = features[:, 1]
    centers_x = features[:, 2]
    volumes = features[:, 3]
    eq_diams = features[:, 4]
    minor_lengths = features[:, 5]
    major_lengths = features[:, 6]
    axis_ratios = features[:, 7]

    # Save each segmentation
    if return_seg:
        verbose_print(args, f'Saving single-cell segmentations to {args.segmentations}')
        singles = np.asarray([r[1] for r in results])
        np.savez_compressed(args.segmentations, singles)

    # Save CSV containing morphologies for each detected centroid
    data = {'com_z': centers_z,
            'com_y': centers_y,
            'com_x': centers_x,
            'volume': volumes,
            'eq_diam': eq_diams,
            'minor_length': minor_lengths,
            'major_length': major_lengths,
            'axis_ratio': axis_ratios}
    df = pd.DataFrame(data)
    df.to_csv(args.output)
    verbose_print(args, f'Morphological features written to {args.output}')

    verbose_print(args, f'Computing morphologies done!')


def morphology_cli(subparsers):
    morphology_parser = subparsers.add_parser('morphology', help="Measure morphological features of nuclei",
                                              description='Compute morphological features from nuclei segmentation')
    morphology_parser.add_argument('input', help="Path to input nuclei binary segmentation Zarr")
    morphology_parser.add_argument('centroids', help="Path to nuclei centroids numpy array")
    morphology_parser.add_argument('output', help="Path to output morphological features CSV")
    morphology_parser.add_argument('--segmentations', help="Path to output nuclei segmentations numpy array", default=None)
    morphology_parser.add_argument('-w', help="Window size", type=int, nargs='+', default=(8, 25, 25))
    morphology_parser.add_argument('-n', help="Number of workers for segmentation", type=int, default=None)
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
    name_cli(nuclei_subparsers)
    return nuclei_parser


def nuclei_main(args):
    commands_dict = {
        'detect': detect_main,
        'segment': segment_main,
        'fluorescence': fluorescence_main,
        'gate': gate_main,
        'morphology': morphology_main,
        'name': name_main,
    }
    func = commands_dict.get(args.nuclei_command, None)
    if func is None:
        print("Pickle Rick uses nuclei subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'nuclei', '-h'])
    else:
        func(args)


# scout nuclei morphology nuclei_binary_segmentation.zarr/ centroids.npy nuclei_segmentations.npz nuclei_morphologies.csv -v
