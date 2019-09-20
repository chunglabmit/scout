Organoid segmentation
======================

Given images of nuclear staining in TIFF format, SCOUT contains tools for segmenting orgnaoid ventricles as well as
the overall organoid shape (foreground).

Downsampling images
--------------------

Since the segmentation of ventricles and whole-organoids is not a single-cell problem, the nuclear stain images are
first downsampled to pixel width of 4 micron. For example, if the lateral dimensions of the original voxels are
0.67 micron, then a downsampling factor of 6x in x and y would give a 4 micron pixel resolution in each z-slice.
To downsample the preprocessed TIFFs, the following command can be used:

.. code-block:: bash

    scout segment downsample color0_rescaled/ syto_down6x 6 6 -v -t

This command will create a new folder called *syto_down6x* containing TIFF images downsampled 6x in y and
6x in x (the image dimension order is (y, x) for 2D and (z, y, x) for 3D). The flag *-t* specifies that the input
folder contains a series of 2D TIFFs. Next, the downsampled 2D TIFFs can be stacked into a single 3D TIFF:

.. code-block:: bash

    scout segment stack syto_down6x/ syto_down6x.tif -v

This downsampled 3D TIFF of the nuclear staining is the input for ventricle and whole-organoid segmentation.

Ventricle segmentation
-----------------------

SCOUT uses a pre-trained U-Net model to segment ventricles from nuclear stain images at 4 micron pixel resolution.
Note that tensorflow must be installed in the scout conda environment for this to work, and the GPU-accelerated
version will greatly improve performance. If the organoid images are larger than 1024 x 1024, then a GPU with more than
16GB memory may be required is using *tensorflow-gpu*. If this is not available, then the CPU-only version of
tensorflow should also work, but it performance may be slow. To run the ventricle segmentation, the following command
can be used:

.. code-block:: bash

   scout segment ventricle syto_down6x.tif path/to/unet.h5 segment_ventricles.tif -t 0.5 -v

where *path/to/unet.h5* is a path to the U-Net model weights and *segment_ventricles.tif* is the resulting binary
ventricle segmentation image. The *-t 0.5* argument specifies a probability threshold for obtaining a binary
image from the U-Net model prediction.

Foreground segmentation
------------------------

The overall organoid can be segmented by blurring the downsampled nuclear stain and thresholding the result, which
can be done using the following command:

.. code-block:: bash

   scout segment foreground syto_down6x.tif segment_foreground.tif -v -t 0.02 -g 8 4 4

where *segment_foreground.tif* is a newly created 3D binary TIFF image containing the overall organoid shape. The
argument *-g 8 4 4* specifies the amount of gaussian smoothing along each the z, y, and x dimensions, and
the argument *-t 0.02* specifies the level at which to threshold the gaussian smoothed image.
