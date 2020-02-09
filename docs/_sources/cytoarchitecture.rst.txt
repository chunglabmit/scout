Cytoarchitectural analysis
===========================

Given the ventricle segmentation, nuclei phyical coordinates, and cell-type labels, SCOUT's cytoarchitectural analysis
computes radial cell profiles to identify the distribution of cellular organizations in each organoid.

Generate ventricle normals
---------------------------

The binary ventricle segmentation must be converted into a vectorized mesh in order to obtain surface normals that
determine the direction for each radial profile. This can be done using the following command:

.. code-block:: bash

   scout cyto mesh segment_ventricles.tif voxel_size.csv mesh_ventricles.pkl -d 1 6 6 -g 2 -s 2 -v -p

This command creates a new file *mesh_ventricles.pkl*, which is a pickled Python dictionary containing the verticies,
normals, and faces of the ventricle mesh.

The argument *-d 1 6 6* specifies the downsampling factor used to generate
*segment_ventricles.tif*. The arguments *-g 2* and *-s 2* specify how much gaussian smoothing should be applied to the
binary image before meshing and what the mesh spacing should be (minimum is 1, which gives the finest mesh).

A 3D render of the ventricle mesh will be shown.
**Note:** The plotting flag *-p* here will not work if mayavi has not been properly installed.

Compute radial cell profiles
-----------------------------

Given the ventricle mesh, the nuclei point cloud can be binned into radially oriented cylinders along each
normal vector. To compute these radial profiles, the following command can be used:

.. code-block:: bash

   scout cyto profiles mesh_ventricles.pkl centroids_um.npy nuclei_gating.npy cyto_profiles.npy -v -p

This command creates a new numpy array *cyto_profiles.npy* containing the radial profiles for each cell-type along
each normal.

**Note:** This command defaults to using all CPU cores available. If the calculation is still too slow, try increasing
the mesh spacing *-s* to reduce the number of profiles to compute.

Setting up clustering analyses
-------------------------------

Due to the stochastic nature of organoid culture, it is difficult to assign distinct semantic labels to each
radial cell profile corresponding to canonical cytoarchitectures. Instead, SCOUT aims to quantify cytoarchitectural
composition of each organoid relative to what is observed in a group of organoids. To accomplish this, SCOUT uses
hierarchical clustering of radial cell profiles to assign cytoarchitectural labels to all organoids in a particular
study.

Since the results of the clustering analysis depend on which organoids are included, it is useful to aggregate
organoid datasets into a separate folder to make this dependency on a group of organoids explicit. We first select
which datasets we would like to include in our analysis folder using the following command:

.. code-block:: bash

    scout multiscale select /path/to/datasets/summary.csv analysis.csv Arlotta_d34 Lancaster_d35 -v


This command reads the summary.csv file, pulls out the paths to the Arlotta_d34 and Lancaster_d35 organoid datasets, and saves the
selected organoids to an new analysis.csv file in the current folder (which is assumed to be analysis folder for Arlotta_d34_vs_Lancaster_d35 comparison based on the command above). Note that this analysis.csv file can be modified to
include more datasets or exclude some datasets from the analysis. The analysis folder can be setup with links to
underlying dataset folders using the following command:

.. code-block:: bash

    scout multiscale setup analysis.csv /path/to/datasets/ -v

This command creates new folders for each dataset in the analysis containing a symbolic link to the original dataset
folder in `/path/to/datasets`. This allows convenient access to the single-cell analysis results and, more importantly,
shares the previous results across multiple analyses for consistency.

Using the example in the **Preprocessing** section, running this command in an `analysis` folder next to the `datasets`
folder would result in the following structure:

.. code-block:: bash

    datasets/
    |   20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    |   20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/
    |   .,, possibly more datasets
    |   summary.csv
    
    analysis_name/
    | analysis.csv
    | Group1/   
    |   20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   |   dataset -> ../../datasets/20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    |   |   dataset -> ../../datasets/20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    | Group2/
    |   20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   |   dataset -> ../../datasets/20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/
    |   |   dataset -> ../../datasets/20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/

Clustering sampled profiles
----------------------------

In order to identify different types of cytoarchitectures in an unbiased manner, the radial cell profiles can be
clustered into groups using hierarchical clustering.

In comparative studies, some organoids may contain completely different cytoarchitectures depending on the model.
For this reason, the clustering analysis must contain representative radial profiles from each organoid.

Given that it is too computationally expensive to concatenate all profiles together and cluster them directly,
we instead randomly sample an equal number of profiles from each organoid in the analysis and cluster based on that
subset. The profiles can be sampled using the following command:

.. code-block:: bash

   scout cyto sample 5000 cyto_sample_index.npy -i cyto_profiles.npy -o cyto_profiles_sample.npy -v

This command will randomly sample 5000 profiles from *cyto_profiles.npy* and store them in *cyto_profiles_sample.npy*.
Another output is *cyto_samply_index.npy*, which contains the index into *cyto_profiles.npy* for each of the sampled
profiles. This command should be run for each organoid, and then the sampled profiles can be combined:

.. code-block:: bash

    scout cyto combine analysis.csv -o cyto_profiles_combined.npy -s cyto_profiles_combined_samples.npy -v

When this command is run, it is meant to find individual cyto_profiles_sample.npy from each subfolder within each group and combine them all. The profiles will be concatenated in order and saved to *cyto_profiles_combined.npy*, along with a new array
*cyto_profiles_combined_samples.npy* which contains integer labels for each profile corresponding to the which
organoid the profile is from. If this process is run correctly, the analysis folder should resemble the construct below. 

.. code-block:: bash
    
    analysis/
    | analysis.csv
    | cyto_profiles_combined.npy
    | cyto_profiles_combined_sample.npy 
    | Group1/   
    |   20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   |   dataset -> ../../datasets/20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    |   |   dataset -> ../../datasets/20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    | Group2/
    |   20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   |   dataset -> ../../datasets/20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/
    |   |   dataset -> ../../datasets/20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/

To perform the cytoarchitecture clustering anb visualization, use the Jupyter notebook
"*determine cyto clusters.ipynb*".

Once the cytoarchitecture clusters have been determined, they can be named using the following command:

.. code-block:: bash

   scout cyto name name1 name2 (...) -o cyto_names.csv -v 

Next Step: 

.. code-block:: bash
   
   scp -r cyto_names.csv /Group1/each_organoid_folder 

Classifying cytoarchitectures
------------------------------

Once the clusters labels have been identified, then all radial profiles can be classified based on those
cytoarchitecture assignments. Please go into the individual organoid subfolder and run the command given below:

.. code-block:: bash

    scout cyto classify ../../cyto_profiles_combined.npy ../../cyto_labels_combined.npy dataset/cyto_profiles.npy cyto_labels.npy -v --umap ../../model_name.umap 

This command uses the combined profiles and cluster labels as a training set to classify all profiles in
*cyto_profiles.npy* using a nearest neighbor classifier. The resulting cytoarchitecture labels are saved to
*cyto_labels.npy*, and the argument *--umap model.umap* specifies which pre-trained UMAP model to embed the
profiles before classification.

3D rendering with Blender
--------------------------

Using Blender 2.8, the following scripts can be used to render the ventricles colored by cytoarchitectural labels
each mesh face as well as the nuclei point clouds.

Export as OBJ and CSV
**********************

See the Jupyter notebook "*Export mesh and points as OBJ*". OBJ files can be imported directly into Blender.
The notebook converts the nuclei physical coordinates from a numpy array into a CSV array that can be read
into Blender using pure Python.

The cytoarchitecture labels correspond to each vertex, but meshes are easier to color by faces in Blender, so the
notebook also uses the vertex-based labels to label each face. The resulting face labels are written to CSV
so that they can also be loaded into Blender using pure Python.

Blender script
***************

In Blender, the following script creates a new material for each unique cytoarchitecture and assigns each face
in the ventricle mesh to the corresponding material.

.. code-block:: python

    import bpy
    import csv

    # Path to face labels
    labels_csv = 'face_labels.csv'

    def read_csv(path):
        with open(path, mode='r') as f:
            line = f.readline().split('\n')[0]
        return line.split(',')

    # Load face labels
    labels = read_csv(labels_csv)
    classes = list(set(labels))
    classes.sort()
    n_classes = len(classes)
    print(f'Read {len(labels)} face labels belonging to {n_classes} classes')

    # Make materials for each class
    context = bpy.context
    obj = context.object
    mesh = obj.data

    existing_material_names = [m.name for m in mesh.materials]
    class_material_names = []
    class_material_index = []
    for i in range(n_classes):
        material_name = f'class {i} material'
        class_material_names.append(material_name)
        if material_name in existing_material_names:
            class_material_index.append(existing_material_names.index(material_name))
        else:
            class_material_index.append(len(mesh.materials))
            mesh.materials.append(bpy.data.materials.new(material_name))
    label_to_index = dict(zip(range(n_classes), class_material_index))

    # Assign faces to materials based on labels
    for f, lbl in zip(mesh.polygons, labels):  # iterate over faces
        print(lbl)
        f.material_index = label_to_index[int(lbl)]
        print("face", f.index, "material_index", f.material_index)
        slot = obj.material_slots[f.material_index]
        mat = slot.material
        if mat is not None:
            print(mat.name)
            print(mat.diffuse_color)
        else:
            print("No mat in slot", f.material_index)

