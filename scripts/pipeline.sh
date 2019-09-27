#scout preprocess histogram Ex_0_Em_0_stitched/ Ex0_hist.csv -s 50 -v
#scout preprocess histogram Ex_1_Em_1_stitched/ Ex1_hist.csv -s 50 -v
#scout preprocess histogram Ex_2_Em_2_stitched/ Ex2_hist.csv -s 50 -v
#scout preprocess rescale Ex_0_Em_0_stitched/ Ex0_hist.csv Ex0_rescaled -t 120 -p 99.7 -v
#scout preprocess rescale Ex_1_Em_1_stitched/ Ex1_hist.csv Ex1_rescaled -t 100 -p 99.7 -v
#scout preprocess rescale Ex_2_Em_2_stitched/ Ex2_hist.csv Ex2_rescaled -t 100 -p 99.7 -v
#scout preprocess convert Ex0_rescaled/ syto.zarr -v -n 8
#scout preprocess convert Ex1_rescaled/ sox2.zarr -v -n 8
#scout preprocess convert Ex2_rescaled/ tbr1.zarr -v -n 8

scout nuclei detect syto.zarr nuclei_probability.zarr centroids.npy --voxel-size voxel_size.csv --output-um centroids_um.npy -v
scout nuclei segment nuclei_probability.zarr centroids.npy nuclei_foreground.zarr nuclei_binary.zarr -v
scout nuclei morphology nuclei_binary.zarr centroids.npy nuclei_morphologies.csv -v
scout nuclei fluorescence centroids.npy nuclei_fluorescence sox2.zarr/ tbr1.zarr/ -v
scout nuclei gate nuclei_fluorescence/nuclei_mfis.npy nuclei_gating.npy 0.1 0.1 -p -v -r 1.5 1.5
scout nuclei name sox2 tbr1 dn -o celltype_names.csv -v

scout niche proximity centroids_um.npy nuclei_gating.npy niche_proximities.npy -r 25 25 -v -k 2 -p
scout niche gate niche_proximities.npy niche_labels.npy --low 0.35 0.30 --high 0.66 0.63 -v --alpha 0.01 -p
scout niche name DN SOX2 TBR1 DP MidTBR1 MidSOX2 MidInter -o niche_names.csv -v

scout segment downsample Ex0_rescaled/ syto_down6x 6 6 -v -t
scout segment stack syto_down6x/ syto_down6x.tif -v
scout segment ventricle syto_down6x.tif /data/datasets/ventricle_segmentation/unet_weights3_zika.h5 segment_ventricles.tif -t 0.5 -v
scout segment foreground syto_down6x.tif segment_foreground.tif -v -t 0.02 -g 8 4 4

scout cyto mesh segment_ventricles.tif voxel_size.csv mesh_ventricles.pkl -d 1 6 6 -g 2 -s 2 -v -p
scout cyto profiles mesh_ventricles.pkl centroids_um.npy nuclei_gating.npy cyto_profiles.npy -v -p
scout cyto sample 5000 cyto_sample_index.npy -i cyto_profiles.npy -o cyto_profiles_sample.npy -v

scout cyto classify cyto_profiles_combined.npy cyto_labels_combined.npy cyto_profiles.npy cyto_labels.npy -v --umap model.umap
scout cyto name name1 name2 ... -o cyto_names.csv -v

scout multiscale features . -d 1 6 6 -v
scout multiscale combine ... --output combined_features.xlsx -v