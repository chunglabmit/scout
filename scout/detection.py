"""
Detection Module
=================

The detection module is responsible for 3D nuclei centroid detection, which is required for single-cell analysis.

"""

from functools import partial
import numpy as np
from skimage import img_as_float32
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from scipy.special import expit
from scout.preprocess import remove_background
from scout import utils
from scout import curvature


# Probability map calculations


def smooth(image, sigma):
    if image.dtype != 'float32':
        image = img_as_float32(image)
    g = gaussian(image, sigma=sigma, preserve_range=True)
    return g


def sigmoid(x):
    return expit(x)


def curvature_probability(eigvals, steepness, offset):
    """Calculate the interest point probability based on 3D curvature eigenvalues

    Parameters
    ----------
    eigvals : ndarray
        4D array of curvature eigenvalues
    steepness : float
        Slope of the logistic function. Larger gives sharper transition between nuclei and background
    offset : float
        Translation of the logistic function. Larger biases towards more negative curvatures

    Returns
    -------
    prob : ndarray
        Curvature interest point probability

    """
    p0 = sigmoid(-steepness * (eigvals[..., 0] + offset))
    p1 = sigmoid(-steepness * (eigvals[..., 1] + offset))
    p2 = sigmoid(-steepness * (eigvals[..., 2] + offset))
    return p0 * p1 * p2


def intensity_probability(image, I0=None, stdev=None):
    """Calculate the foreground probability using exponential distribution

    Parameters
    ----------
    image : ndarray
        Input image
    I0 : float, optional
        Normalization value. Default, mean of image
    stdev : float, optional
        Width of the transition to foreground. Default, stdev of normalized image

    Returns
    -------
    prob : ndarray
        Foreground probability map

    """
    if I0 is None:
        I0 = image.mean()
    normalized = image / I0
    if stdev is None:
        stdev = normalized.std()
    return 1 - np.exp(-normalized ** 2 / (2 * stdev ** 2))


def nucleus_probability(image, sigma, steepness=500, offset=0.0005, I0=None, stdev=None):
    """Calculate the nucleus probability map using logistic regression over curvature eigenvalues

    Parameters
    ----------
    image : ndarray
        3D image volume of nuclei staining
    sigma : int or tuple
        Amount to blur the image before processing
    steepness : float
        Slope of the logistic function. Larger gives sharper transition between nuclei and background
    offset : float
        Translation of the logistic function. Larger biases towards more negative curvatures

    Returns
    -------
    prob : ndarray
        Nuclei probability map

    """
    if sigma is not None:
        g = smooth(image, sigma)
    else:
        g = image
    eigvals = curvature.eigvals_of_weingarten(g)
    p_curvature = curvature_probability(eigvals, steepness, offset)
    p_intensity = intensity_probability(g, I0, stdev)
    return p_curvature * p_intensity


def nuclei_centers_probability(prob, threshold, min_dist):
    """
    Find nuclei centroids with a minimum probability at least `min_dist` apart

    Parameters
    ----------
    prob : ndarray
        Input probability image
    threshold : float
        Minimum probability allowed for a centroid
    min_dist : int
        Minimum pixel distance allowed between detected centroids

    Returns
    -------
    output : ndarray
        Coordinates of detected centroids

    """
    prob = remove_background(prob, threshold)
    return peak_local_max(prob, min_distance=min_dist, threshold_abs=threshold)


def _detect_nuclei_chunk(input_tuple, overlap, sigma, min_intensity, steepness, offset, I0=None, stdev=None, prob_thresh=0.5, min_dist=1, prob_output=None):
    """
    Detect nuclei centroids from a chunk of a Zarr array

    Parameters
    ----------
    input_tuple : tuple
    overlap : int
    sigma : int, tuple
    min_intensity : float
    steepness : float
    offset : float
    I0 : float
    stdev : float
    prob_thresh : float
    min_dist : int
    prob_output : Zarr Array

    Returns
    -------
    centers : ndarray

    """
    arr, start_coord, chunks = input_tuple

    ghosted_chunk, start_ghosted, _ = utils.extract_ghosted_chunk(arr, start_coord, chunks, overlap)

    if ghosted_chunk.max() < min_intensity:
        return None

    prob = nucleus_probability(ghosted_chunk, sigma, steepness, offset, I0, stdev)

    if prob_output is not None:
        start_local = start_coord - start_ghosted
        stop_local = np.minimum(start_local + np.asarray(chunks),
                                np.asarray(ghosted_chunk.shape))
        prob_valid = utils.extract_box(prob, start_local, stop_local)
        stop_coord = start_coord + np.asarray(prob_valid.shape)
        utils.insert_box(prob_output, start_coord, stop_coord, prob_valid)

    centers_local = nuclei_centers_probability(prob, prob_thresh, min_dist)

    if centers_local.size == 0:
        return None

    # Filter out any centers detected in ghosted area
    centers_interior = utils.filter_ghosted_points(start_ghosted, start_coord, centers_local, chunks, overlap)

    # change to global coordinates
    centers = centers_interior + start_ghosted
    return centers


def _filter_nones(centers_list):
    """
    Filters out `None` from input list

    Parameters
    ----------
    centers_list : list
        List potentially containing `None` elements

    Returns
    -------
    new_list : list
        List without any `None` elements

    """
    return [c for c in centers_list if c is not None]


def detect_nuclei_parallel(z_arr, sigma, min_intensity, steepness, offset, I0, stdev, prob_thresh, min_dist, chunks, overlap, nb_workers=None, prob_output=None):
    """
    Detect nuclei centroids from a chunked Zarr array with parallel processing

    Parameters
    ----------
    z_arr : Zarr Array
        Input chunked Zarr array
    sigma : int, tuple
        Amount of blurring
    min_intensity : float
        Minimum intensity allowed for centroid
    steepness : float
        Sensitivity of the curvature-based probability map
    offset : float
        Bias of the curvature-based probability map
    I0 : float
        Reference intensity for image intensity prior. Can typically be set using Otsu's method.
    stdev : float
        Sensitivity for the image intensity prior. Can typically be set to the standard deviation of the image.
    prob_thresh : float
        Minimum probability allowed for centroid
    min_dist : int
        Minimum distance allowed between detected centroids
    chunks : tuple
        Chunk size to process in parallel. This must match z_arr.chunks if using GPU curvature computations.
    overlap : int
        Amount of overlap between adjacent chunks
    nb_workers : int
        Number of parallel processes to use
    prob_output : Zarr Array
        Output array for intermediate probability map used for centroid detection

    Returns
    -------
    centers : ndarray

    """
    f = partial(_detect_nuclei_chunk,
                overlap=overlap,
                sigma=sigma,
                min_intensity=min_intensity,
                steepness=steepness,
                offset=offset,
                I0=I0,
                stdev=stdev,
                prob_thresh=prob_thresh,
                min_dist=min_dist,
                prob_output=prob_output)
    results = utils.pmap_chunks(f, z_arr, chunks, nb_workers, use_imap=True)
    return np.vstack(_filter_nones(results))
