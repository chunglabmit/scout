Multiscale statistical analysis
=================================

Based on the single-cell, proximity, segmentation, and cytoarchitectural results, multiscale
features can be calculated and combined into a hyperdimensional statistical analysis.

This statistical analysis is best conceptualized as an unbiased hypothesis generation tool.
That is, it contains a more unbiased view of whole organoids compared to section-based analyses, and
researchers can always go back to the image data to ask new questions inspired by this statistical analysis.

Aggregating features
---------------------

First, the predetermined set of multiscale features can be calculated using the following command inside an
organoid group dataset folder. Please "cd" into groupN (for example, Lancaster_d35) and execute the following:

.. code-block:: bash

   scout multiscale features organoid_folder_name(usually 2019...)/. -d 1 6 6 -v

This command takes the current folder (specified by ".") and looks for intermediate results, including
*centroids_um.npy* and *cyto_labels.npy*. The argument *-d 1 6 6* specifies the z, y, and x downsampling factors
used for the ventricle segmentation. This command should be repeated for each organoid in the study (the same
organoids sampled from to generate the cytoarchitecture clusters). This command will create an Excel spreadsheet
in the current folder called *organoid_features.xlsx*.

The multiscale features for each organoid can be combined into a single Excel table using the following command:

.. code-block:: bash

   scout multiscale combine path/to/org1 path/to/org2 (...) --output combined_features.xlsx -v 

An easier way to get this done without inputing specific paths is by using analysis.csv file as shown in the command below:

.. code-block:: bash

   scout multiscale combine analysis.csv --output combined_features.xlsx -v 


This command expects the *organoid_features.xlsx* to be present in each organoid folder. The combined table is written
to *combined_features.xlsx*.

Statistical testing
--------------------

To perform statistical tests on the combined features, use the notebook called "*T-tests and volcano plots.ipynb*".
