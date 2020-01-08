Preprocessing
=============

Volumetric image inputs
-------------------------

The SCOUT pipeline assume that a series of 2D TIFF images are given to form 3D volumes.
Each image should correspond to an optical z-section in ascending alphanumeric order.
Multiple channels are stored in separate folders containing the same number of 2D TIFF images.

For example, for a 3-channel volumetric image containing 1000 z-planes, the starting data may be organized as follows:

| dataset_folder/
|   **color_0/**
|       *000.tif*
|       *...*
|       *999.tif*
|   **color_1/**
|       *000.tif*
|       *...*
|       *999.tif*
|   **color_2/**
|       *000.tif*
|       *...*
|       *999.tif*

**Note:** The exact names of the files and folders do not matter--SCOUT automatically finds all TIFFs and sorts them
by name. However, each channel must contain the same number of images.

Organoid dataset organization
------------------------------

It is customary to place all dataset folders under a single `datasets` directory with an accompanying `summary.csv`
file that describes what each dataset contains. `summary.csv` should contain a dataframe indexed by the
dataset `path`, which is the name of the dataset_folder, and a field called `type`, which contains a string describing
the organoid type or experimental group.

As an example, this is a `summary.csv` containing 2 Lancaster_d35 organoids and 2 Lancaster_d60 organoids.

.. code-block:: bash

    path,type
    20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60,Lancaster_d35
    20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60,Lancaster_d35
    20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50,Lancaster_d60
    20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60,Lancaster_d60

The accompanying `datasets` folder would look like this:

.. code-block:: bash

    datasets/
    |   20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    |   20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/
    |   summary.csv


Image normalization
--------------------

In order to make downstream processing more consistent, the first step is to normalize each channel. Rather than
normalizing each z-slice independently, which would introduce irregular brightness artifacts along the z-axis, it
is better to base the normalization on the histogram of the entire 3D image. Since the exact histogram may take a
long time to compute, we instead can select a few equally-spaced z-slices to estimate it:

.. code-block:: bash

   scout preprocess histogram color_0/ color0_hist.csv -s 50 -v

The resulting histogram will be stored in the *color0_hist.csv* CSV file.
The *-s 50* argument means "take every 50th image" to estimate the histogram. The *-v* argument is a verbose flag.
This command should be repeated for each channel.

Given the stack histogram, we can choose to normalize to any percentile. For example, if we wanted to scale the image
so that the maximum value is 1.0, then we would normalize to the 100th percentile (the maximum value). Due to imaging
noise, it is sometimes more robust to instead normalize based on a slightly lower percentile:

.. code-block:: bash

   scout preprocess rescale color_0/ color0_hist.csv color0_rescaled -t 120 -p 99.7 -v

This will rescale each image in *color_0* based on the 99.7th percentile of *color0_hist.csv* and save the results
in a new folder called *color0_rescaled*. The *-t 120* argument specifies the grayscale intensity of the
background, which gets subtracted from each image before normalization.
This command should be repeated for each channel.

Convert to Zarr format
-----------------------

To facilitate parallel processing of volumetric datasets, the normalized TIFF stacks need to be
broken into more manageable chunks. Fortunately, Zarr is a Python package that provides a chunk-compressed array
data structure that we can use. To convert the TIFFs into a Zarr array, use the following command:

.. code-block:: bash

    scout preprocess convert color0_rescaled/ syto.zarr -v -n 4

This will convert the TIFF stack in *color0_rescaled* into a Zarr NestedDirectoryStore called *syto.zarr*.
The *-n 4* argument specifies how many parallel workers to use in this conversion process.
This command should be repeated for each channel.

**Note:** Using too many workers to convert TIFFs to Zarr may result in error messages about folder creation.
A pull request has been submitted with Zarr regarding this issue.

