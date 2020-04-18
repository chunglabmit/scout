Walkthrough with test data
==========================

The following walkthrough illustrates how to use SCOUT on a small organoid
dataset using Docker. Docker provides a cross-platform way to run SCOUT with
minimal setup. The test dataset contains stitched (whole-organoid) syto16, SOX2,
and TBR1 images from a portion of a d35 cerebral organoid cultured similar to
the Lancaster protocol (described in the Methods section).

Docker setup
-------------

The easiest way to ensure a similar runtime environment for SCOUT on Windows,
Mac, and Linux is by using the :code:`chunglabmit/scout` Docker image hosted on
:code:`Dockerhub`. First, you will need to install "Docker Desktop" (which is
free) by following the platform-specific instructions at
https://docs.docker.com/get-docker/

Once installed, you will need to download the pre-built Docker image for SCOUT.
Open a terminal and use :code:`docker pull` to download it:

.. code-block:: bash

    sudo docker pull jswaney/scout

This image may be moved to :code:`chunglabmit/scout` in the future. Note that
the `sudo` keyword may not be needed on your platform for Docker commands.

Docker Desktop on Windows and Mac may restrict the amount of CPU and RAM
resources that each container can use by default. You can adjust resource
allocation by accessing the Docker settings through the Docker tray icon. You
may also need to allow Docker access to other drives (D:, E:, etc) if the
:code:`scout-data` directory is placed on a different drive. Lastly, if you want
to run Jupyter notebooks with the SCOUT Docker image, you may need to follow
some platform-specific networking setup (port forwarding, routing, etc), which
you can read more about at
https://docs.docker.com/docker-for-windows/networking/ or
https://docs.docker.com/docker-for-mac/networking/ for Windows and Mac,
respectively.


Download test data
-------------------

After installing the SCOUT Docker image, a small test dataset (~3 GB) can be
downloaded from Dropbox. The test dataset is distributed as an archive called
:code:`scout-data.zip`, which contains two subfolders: :code:`test` and
:code:`results`. The :code:`test` folder contains all the data needed to start
the SCOUT analysis from the begining, such as raw stitched images from a
microscope. The :code:`results` folder contains all the intermediate results
expected from completing the following walkthrough. This data is included for
completeness and verification purposes and is not required to actually run SCOUT
on newly acquired data.

First, download the test dataset from
https://www.dropbox.com/s/j37p5m7q7qk1mp1/scout-data.zip?dl=0 and unzip it. Make
note of the resulting :code:`scout-data/test` and :code:`scout-data/results`
folders.

Open a terminal and move into the :code:`scout-data/test` directory:

.. code-block:: bash

    cd scout-data/test

This folder will be mounted into the SCOUT Docker container using the :code:`-v
$(pwd):/scout/data` argument to :code:`docker run` throughout the following
walkthrough.

Preprocessing
-------------

The first step in the SCOUT pipeline is to estimate the overall image histograms
for each channel.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess histogram \
        data/dataset/color_0/ data/dataset/color0_hist.csv -s 1 -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess histogram \
        data/dataset/color_1/ data/dataset/color1_hist.csv -s 1 -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess histogram \
        data/dataset/color_2/ data/dataset/color2_hist.csv -s 1 -v

Using these histograms, we can normalize the images to the range [0, 1] and
apply a background threshold.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess rescale \
        data/dataset/color_0/ data/dataset/color0_hist.csv data/dataset/color0_rescaled \
        -t 120 -p 99.7 -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess rescale \
        data/dataset/color_1/ data/dataset/color1_hist.csv data/dataset/color1_rescaled \
        -t 100 -p 99.7 -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess rescale \
        data/dataset/color_2/ data/dataset/color2_hist.csv data/dataset/color2_rescaled \
        -t 100 -p 99.7 -v

In order to more easily work with volumetric image data, we the convert the 2D
TIFF stacks into 3D Zarr arrays. Each Zarr array is a nested folder of chunk
compressed voxel data. By default, the chunk size is (64, 64, 64). 

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess convert \
        data/dataset/color0_rescaled data/dataset/syto.zarr -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess convert \
        data/dataset/color1_rescaled data/dataset/sox2.zarr -v
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout preprocess convert \
        data/dataset/color2_rescaled data/dataset/tbr1.zarr -v


Nuclei Detection
-----------------

Once we have the syto16.zarr array, we can detect nuclei centroids using
parallel processing on each image chunk. Note that the current Docker image does
not support GPU acceleration, and this step would be much faster by installing
from source on a machine with a GPU.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei detect data/dataset/syto.zarr \
        data/dataset/nuclei_probability.zarr data/dataset/centroids.npy \
        --voxel-size data/dataset/voxel_size.csv \
        --output-um data/dataset/centroids_um.npy -n 2 -v

Given these nuclei centroids, we can perform a seeded watershed segmentation of
the nuclei probability array to obtain the shape of each detected nucleus. This
operation is done with some overlap between adjacent chunks to avoid artifacts
at the boundaries between adjacent chunks in the watershed lines.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei segment \
        data/dataset/nuclei_probability.zarr data/dataset/centroids.npy \
        data/dataset/nuclei_foreground.zarr data/dataset/nuclei_binary.zarr -n 2 -v

Given this binary nuclei segmentation, we can compute morphological features for
each nucleus. The resulting morphological features are stored in a CSV.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei morphology \
        data/dataset/nuclei_binary.zarr data/dataset/centroids.npy \
        data/dataset/nuclei_morphologies.csv -v

Finally, we can sample the fluorescence in the other channels (SOX2 and TBR1 in
this case) at each nucleus centroid. 

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei fluorescence \
        data/dataset/centroids.npy data/dataset/nuclei_fluorescence/ \
        data/dataset/sox2.zarr/ data/dataset/tbr1.zarr/ -v

The resulting mean fluorescence intensities
(MFIs) are useful for gating cells into different cell types based on protein
expression.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei gate \
        data/dataset/nuclei_fluorescence/nuclei_mfis.npy \
        data/dataset/nuclei_gating.npy 0.35 0.25 -v

In this case, high SOX2 expression is used to identify neural progenitors and
high TBR1 expression is used to identify post-motitic neurons. Cells that have
low SOX2 and TBR1 expression are called "double negative" (DN). Cell types can
be named in order using the following command:

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout nuclei name \
        sox2 tbr1 dn -o data/dataset/celltype_names.csv -v


Microenvironment Analysis
--------------------------

(Note that this was formerly called `niche` analysis)

Given nuclei centroids and cell type labels, we can further describe the
microenvironment around each cell. To do this, we compute the `proximity` to
each of the non-DN cell types, which is described in the Method section.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout niche proximity \
        data/dataset/centroids_um.npy data/dataset/nuclei_gating.npy \
        data/dataset/niche_proximities.npy -r 25 25 -k 2 -v

These spatial proximities are attibutes of each cell describing the local
environment. The next step is to use these proximity values to further gate
cells into subpopulations based on their spatial context.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout niche gate \
        data/dataset/niche_proximities.npy data/dataset/niche_labels.npy \
        --low 0.2 0.2 --high 0.66 0.63 -v

Here, we defined a `low` and high` proximity threshold for SOX2 and TBR1
separately. This results in 7 subpopulations (3 high, 3 mid, and 1 low), which
can be named using the following command:

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout niche name \
        DN SOX2 TBR1 DP MidTBR1 MidSOX2 MidInter -o data/dataset/niche_names.csv -v


Ventricle Segmentation
-----------------------

Next, we turn to ventricle segmentation, which is required to calculate radial
profiles in a cytoarchitecture analysis. The pretrained U-Net model assumes that
each input image is of nuclear staining at 4 um pixel resoltion. We, therefore,
resize the normalized nuclei images and stack them into a single 3D TIFF.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout segment downsample \
        data/dataset/color0_rescaled/ data/dataset/syto_down6x 6 6 -v -t
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout segment stack \
        data/dataset/syto_down6x/ data/dataset/syto_down6x.tif -v

This 3D TIFF can be passed to the U-Net model for ventricle segmentation, which
occurs one 2D slide at a time.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout segment ventricle \
        data/dataset/syto_down6x.tif models/unet_weights3_zika.h5 \
        data/dataset/segment_ventricles.tif -t 0.5 -v

We also need a foreground segmentation to determine the overall organoid size
and shape. A foreground segmentation can be computed by thresholding.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout segment foreground \
        data/dataset/syto_down6x.tif data/dataset/segment_foreground.tif -v -t 0.02 -g 8 4 4


Cytoarchitecture Analysis
--------------------------

Given the ventricle segmentation, nuclei centroids, and cell types labels,
radial profiles from each ventricle can be computed. First, the ventricle
segmentation is turned into a polygon mesh (using the marching cubes algorithm).

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout cyto mesh \
        data/dataset/segment_ventricles.tif data/dataset/voxel_size.csv \
        data/dataset/mesh_ventricles.pkl -d 1 6 6 -g 2 -s 3 -v

Then, normal vectors from this mesh are used to compute radial profiles for each
cell type.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout cyto profiles \
        data/dataset/mesh_ventricles.pkl data/dataset/centroids_um.npy \
        data/dataset/nuclei_gating.npy data/dataset/cyto_profiles.npy -v

Finally, we randomly sample from the large number of radial profiles to be able
to cluster radial profiles across many organoids. This step isn't required in
this case, but we include it for completeness.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout cyto sample 5000 \
        data/dataset/cyto_sample_index.npy -i data/dataset/cyto_profiles.npy \
        -o data/dataset/cyto_profiles_sample.npy -v


Then, we would compute clusters of cytoarchitectures across all organoids by
combining sampled profiles and using the :code:`determine cyto clusters.ipynb`
notebook. You can access and use these notebooks by starting a Jupyter server
within the SCOUT Docker container:

.. code-block:: bash

    sudo docker run -it -v "$(pwd):/scout/data" -p 8888:8888 chunglabmit/scout jupyter --ip 0.0.0.0


Note that the positions of the `-p` and `--ip` arguments are important because
`-p` is for Docker port forwarding and `--ip` is for the Jupyter server. You can
navigate to :code:`localhost:8888` in your browser and copy the access token
printed to the terminal as :code:`/?token={copy-this-text}`.

For the sake of brevity, we simply provide precomputed profiles, labels, and a
fit UMAP model from our d35/d60 comparison. With these, we can classify the
cytoarchitecture of all radial profiles.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout cyto classify \
        data/cyto_profiles_combined.npy data/cyto_labels_combined.npy \
        data/dataset/cyto_profiles.npy data/cyto_labels.npy -v \
        --umap data/model_d35_d60.umap

Note that because the test dataset is not a full 3D dataset, the resulting
radial profiles and cytoarchitecture labels obtained here may have some
artifacts due to empty profiles near the top and bottom of the test volume.

We can provide appropriate names for each cytoarchitecture cluster after
inspecting each cluster in the :code:`determine cyto clusters.ipynb` notebook.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout cyto name \
    TBR1-LowDN TBR1-HighDN Surface Artifacts DN Adjacent -o data/cyto_names.csv -v


Multiscale Analysis
--------------------

All of the intermediate results above are used to compute multiscale features
for each dataset in an analysis. Note that the following command assumes that
the intermediate results are named as shown in the previous steps.

.. code-block:: bash

    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout multiscale features data/ \
        -d 1 6 6 -g 2 -v

This command will create an Excel file called :code:`organoid_features.xlsx`, which
can be combined as follows:

.. code-block:: bash

    # Just an example (won't work without multiple organoid datasets)
    sudo docker run -v "$(pwd):/scout/data" chunglabmit/scout multiscale combine \
       dataset1/ dataset2/ ... --output combined_features.xlsx -v

These combined organoid features are used to run statistical tests in the
:code:`T-tests and volcano plots.ipynb` notebook.
