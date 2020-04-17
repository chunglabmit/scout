Walkthrough with test data
==========================

Docker setup
-------------

Some tips on Docker here

Download test data
-------------------

Link to Dropbox here

Move to the :code:`scout-data/test` directory:

.. code-block:: bash

    cd scout-data/test

This folder will be mounted into the SCOUT Docker container using the :code:`-v
$(pwd):/scout/data` argument to :code:`docker run` throughout the following
walkthrough.

Preprocess
-----------

Calculate histograms for each channel

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess histogram \
        data/dataset/color_0/ data/dataset/color0_hist.csv -s 1 -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess histogram \
        data/dataset/color_1/ data/dataset/color1_hist.csv -s 1 -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess histogram \
        data/dataset/color_2/ data/dataset/color2_hist.csv -s 1 -v


Rescale images to [0, 1] range

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess rescale \
        data/dataset/color_0/ data/dataset/color0_hist.csv data/dataset/color0_rescaled \
        -t 120 -p 99.7 -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess rescale \
        data/dataset/color_1/ data/dataset/color1_hist.csv data/dataset/color1_rescaled \
        -t 100 -p 99.7 -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess rescale \
        data/dataset/color_2/ data/dataset/color2_hist.csv data/dataset/color2_rescaled \
        -t 100 -p 99.7 -v


Convert 2D TIFFs to 3D Zarr format

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess convert \
        data/dataset/color0_rescaled data/dataset/syto.zarr -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess convert \
        data/dataset/color1_rescaled data/dataset/sox2.zarr -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout preprocess convert \
        data/dataset/color2_rescaled data/dataset/tbr1.zarr -v


Nuclei
-------

Detect nuclei centroids (Note that this would be much faster if installed from
source with GPU resources)

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei detect data/dataset/syto.zarr \
        data/dataset/nuclei_probability.zarr data/dataset/centroids.npy \
        --voxel-size data/dataset/voxel_size.csv \
        --output-um data/dataset/centroids_um.npy -n 2 -v


Segment individual nuclei segmentation using watershed

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei segment \
        data/dataset/nuclei_probability.zarr data/dataset/centroids.npy \
        data/dataset/nuclei_foreground.zarr data/dataset/nuclei_binary.zarr -n 2 -v


Compute nuclei morphological features

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei morphology \
        data/dataset/nuclei_binary.zarr data/dataset/centroids.npy \
        data/dataset/nuclei_morphologies.csv -v


Sample fluorescence at each nucleus in other channels

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei fluorescence \
        data/dataset/centroids.npy data/dataset/nuclei_fluorescence/ \
        data/dataset/sox2.zarr/ data/dataset/tbr1.zarr/ -v


Gate fluorescence to get cell populations

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei gate \
        data/dataset/nuclei_fluorescence/nuclei_mfis.npy \
        data/dataset/nuclei_gating.npy 0.35 0.25 -v


Give the names to the gated populations

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout nuclei name \
        sox2 tbr1 dn -o data/dataset/celltype_names.csv -v


Microenvironment (formerly called "niche")
-------------------------------------------

Calculate proximities to each cell type

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout niche proximity \
        data/dataset/centroids_um.npy data/dataset/nuclei_gating.npy \
        data/dataset/niche_proximities.npy -r 25 25 -k 2 -v


Gate proximities into subpopulations

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout niche gate \
        data/dataset/niche_proximities.npy data/dataset/niche_labels.npy \
        --low 0.2 0.2 --high 0.66 0.63 -v


Give names to gated subpopulations

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout niche name \
        DN SOX2 TBR1 DP MidTBR1 MidSOX2 MidInter -o data/dataset/niche_names.csv -v


Segment
--------

Resize normalized nuclei images to 4 um pixel width and stack them into a single
3D TIFF

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout segment downsample \
        data/dataset/color0_rescaled/ data/dataset/syto_down6x 6 6 -v -t
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout segment stack \
        data/dataset/syto_down6x/ data/dataset/syto_down6x.tif -v


Segment ventricles using U-Net

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout segment ventricle \
        data/dataset/syto_down6x.tif models/unet_weights3_zika.h5 \
        data/dataset/segment_ventricles.tif -t 0.5 -v


Segment foreground by thresholding

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout segment foreground \
        data/dataset/syto_down6x.tif data/dataset/segment_foreground.tif -v -t 0.02 -g 8 4 4


Cytoarchitecture
----------------

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout cyto mesh \
        data/dataset/segment_ventricles.tif data/dataset/voxel_size.csv \
        data/dataset/mesh_ventricles.pkl -d 1 6 6 -g 2 -s 3 -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout cyto profiles \
        data/dataset/mesh_ventricles.pkl data/dataset/centroids_um.npy \
        data/dataset/nuclei_gating.npy data/dataset/cyto_profiles.npy -v
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout cyto sample 5000 \
        data/dataset/cyto_sample_index.npy -i data/dataset/cyto_profiles.npy \
        -o data/dataset/cyto_profiles_sample.npy -v


Then, compute clusters of cytoarchitectures across all organoids by combining
sampled profiles and using the :code:`determine cyto clusters.ipynb` notebook. You can
access and use these notebooks by starting a Jupyter server within the SCOUT
Docker container:

.. code-block:: bash

    sudo docker run -it -v $(pwd):/scout/data -p 8888:8888 chunglabmit/scout jupyter --ip 0.0.0.0


Note that the positions of the `-p` and `--ip` arguments are important because
`-p` is for Docker port forwarding and `--ip` is for the Jupyter server. You can
navigate to :code:`localhost:8888` in your browser and copy the access token printed
to the terminal as :code:`/?token={copy-this-text}`.

For the sake of brevity, we simply provide precomputed profiles, labels, and a
fit UMAP model from our d35/d60 comparison. With these, we can classify the
cytoarchitecture of all radial profiles.

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout cyto classify \
        data/cyto_profiles_combined.npy data/cyto_labels_combined.npy \
        data/dataset/cyto_profiles.npy data/cyto_labels.npy -v \
        --umap data/model_d35_d60.umap


Given names to the cytoarchitecture classes

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout cyto name \
    TBR1-LowDN TBR1-HighDN Surface Artifacts DN Adjacent -o data/cyto_names.csv -v


Multiscale features
--------------------

All of the intermediate results above are used to compute multiscale features
for each dataset in an analysis

.. code-block:: bash

    sudo docker run -v $(pwd):/scout/data chunglabmit/scout multiscale features data/ \
        -d 1 6 6 -g 2 -v


This command will create an Excel file called :code:`organoid_features.xlsx`, which
can be combined as follows:

.. code-block:: bash

    # Just an example (won't work without multiple organoid datasets)
    sudo docker run -v $(pwd):/scout/data chunglabmit/scout multiscale combine \
       dataset1/ dataset2/ ... --output combined_features.xlsx -v


These combined organoid features are used to run statistical tests in the
:code:`T-tests and volcano plots.ipynb` notebook
