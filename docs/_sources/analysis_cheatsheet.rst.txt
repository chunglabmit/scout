Cheatsheet: "Statistical analysis"
----------------------------------


**Cytoarchitectural Analysis:**


Generate ventricle normals:

.. code-block:: bash

   scout cyto mesh segment_ventricles.tif voxel_size.csv mesh_ventricles.pkl -d 1 6 6 -g 2 -s 2 -v 

Compute radial cell profiles:

.. code-block:: bash

   scout cyto profiles mesh_ventricles.pkl centroids_um.npy nuclei_gating.npy cyto_profiles.npy -v -p

Setting up analysis folders for group comparison:

.. code-block:: bash

    mkdir analysis/Arlotta_d34_vs_Lancaster_d35 
    cd Arlotta_d34_vs_Lancaster_d35

.. code-block:: bash
   
    scout multiscale select ../../datasets/summary.csv analysis.csv Arlotta_d34 Lancaster_d35 -v

.. code-block:: bash

    scout multiscale setup analysis.csv ../../datasets/ -v

Folder structure should look like:

.. code-block:: bash

    datasets/
    |   20190419_14_35_07_AA_org1_488LP13_561LP120_642LP60/
    |   20190419_15_50_16_AA_org2_488LP13_561LP120_642LP60/
    |   20190509_16_55_31_AA-orgs5.8.19_org1_488LP15_561LP140_642LP50/
    |   20190531_14_31_36_AA_org2_488LP13_561LP140_642LP60/
    |   .,, possibly more datasets
    |   summary.csv
    
    analysis/analysis_folder_name/
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

Clustering sampled profiles:
Run this in each individual organoid folders:

.. code-block:: bash

   cd analysis/analysis_folder_name/Group1/2019...(organoid_folder_name)/dataset(symlink) 
   scout cyto sample 5000 cyto_sample_index.npy -i cyto_profiles.npy -o cyto_profiles_sample.npy -v

.. code-block:: bash

   cd ../..
   scout cyto sample 5000 cyto_sample_index.npy -i cyto_profiles.npy -o cyto_profiles_sample.npy -v (repeat this for all organoid folders)

Next, combine all sampled profiles with this command:

.. code-block:: bash

    scout cyto combine analysis.csv -o cyto_profiles_combined.npy -s cyto_profiles_combined_samples.npy -v

.. code-block:: bash
    
    analysis/analysis_folder_name/
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

Perform cytoarchitecture clustering and visualization using Jupyter notebook "**determine cyto clusters.ipynb**".

Next, Send Alex the images to figure out right names for clusters.

Cytoarchitecture clusters naming with help from Alex:

.. code-block:: bash

   scout cyto name name1 name2 (...) -o cyto_names.csv -v 

Next Step: 

.. code-block:: bash
   
   scp -r cyto_names.csv /Group1/each_organoid_folder 

Classifying cytoarchitectures:
Use "*determine cyto clusters.ipynb*" to get a umap model to use the command below: 

.. code-block:: bash
    
    cd analysis/analysis_folder_name/Group1/2019...(organoid_folder_name) 
    scout cyto classify ../../cyto_profiles_combined.npy ../../cyto_labels_combined.npy dataset/cyto_profiles.npy cyto_labels.npy -v --umap ../../model_Group1_and_Group.umap 



Exporting OBJ and CSV (3D rendering with Blender 2.8):

Look into the Jupyter notebook "**Export mesh and points as OBJ**". 
Import OBJ into Blender.

Blender script:

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



**Multiscale statistical analysis:**

Input files needed for the command to work are: **centroids_um.npy** and **cyto_labels.npy** 

"." should specify that these files be inputted. 

.. code-block:: bash
   
   cd analysis/analysis_folder_name/Group1/  
   scout multiscale features organoid_folder_name(usually 2019...)/. -d 1 6 6 -v

If this worked you should see **organoid_features.xlsx** file in the organoid folder.

Combine each **organoid_features.xlsx** from each organoid folder into a cumulative **combined_features.xlsx** file using: 

.. code-block:: bash

   scout multiscale combine analysis.csv --output combined_features.xlsx -v 

Statistical testing:

Use the notebook "**T-tests and volcano plots.ipynb**" for statistical tests on the combined features.
