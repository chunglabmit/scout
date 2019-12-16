Single-cell analysis
=====================

The single-cell analysis requires a volumetric nuclear stain image in Zarr format to find nuclei centroids.
These nuclei centroids are used as seed points in a 3D watershed segmentation, providing morphological features.
The nuclei centroids are also coordinates where to sample other channels for single-cell expression levels.

Name each channel
-----------------

First, it is convenient to provide the name of the staining target in a separate CSV file for building
multiscale features later.

.. code-block:: bash

   scout nuclei name sox2 tbr1 dn -o celltype_names.csv -v

Nuclei detection
----------------

The detection algorithm uses a filter based on curvature in nuclear signal, which is used later for segmentation.
Also, the coordinates of each nucleus in physical space is also needed for future spatial analysis.
These global positions in micron units can only be computed if the user provides the voxel size.

.. code-block:: bash

   scout nuclei detect syto.zarr nuclei_probability.zarr centroids.npy --voxel-size voxel_size.csv --output-um centroids_um.npy -v

where *nuclei_probability.zarr* is a new Zarr array containing the probability of each voxel being a nucleus centroid
and *centroids.npy* is a numpy array containing the voxel coordinates of the detected nuclei. In this case, we are
also passing in *--voxel-size voxel_size.csv* in order to get a new numpy array *--output-um centroids_um.npy*
containing the physical coordinates of the nuclei centroids.

Nuclei segmentation
--------------------

To segment individual nuclei, run the following command:

.. code-block:: bash

   scout nuclei segment nuclei_probability.zarr centroids.npy nuclei_foreground.zarr nuclei_binary.zarr -v

This will create two new Zarr arrays, *nuclei_foreground.zarr* and *nuclei_binary.zarr*. The *nuclei_binary.zarr*
array contains a binary volumetric image where each nucleus as been separated with a watershed line.
This watershed segmentation is used to compute morphological features for each nucleus with the following command:

.. code-block:: bash

   scout nuclei morphology nuclei_binary.zarr centroids.npy nuclei_morphologies.csv -v

The result *nuclei_morphologies.csv* is a CSV file containing a table of morphological features for each detected cell.

Cytometric analysis
--------------------

To sample fluorescence at each nucleus, run the following command:

.. code-block:: bash

   scout nuclei fluorescence centroids.npy nuclei_fluorescence sox2.zarr/ tbr1.zarr/ -v

This command writes a few files in a new folder called *nuclei_fluorescence*, including a table of all mean
fluorescence intensities (MFIs) for each cell. These MFIs can be used to gate cells using the following command:

.. code-block:: bash

   scout nuclei gate nuclei_fluorescence/nuclei_mfis.npy nuclei_gating.npy 0.2 0.1 -p -v -r 1.5 1.5

where *nuclei_gating.npy* is a numpy array containing labels for each cell and gated channel.
In this case, we are gating the first channel (sox2.zarr) at 0.2 and the second channel (tbr1.zarr) at 0.1.
The flag *-p* plots a 2D histogram, and the *-r 1.5 1.5* argument specifies the maximum range of the plot.

