"""
Preprocess Module
==================

This module ...
"""


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
