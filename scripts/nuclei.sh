scout nuclei detect syto.zarr nuclei_probability.zarr centroids.npy --voxel-size voxel_size.csv --output-um centroids_um.npy -v
scout nuclei segment nuclei_probability.zarr centroids.npy nuclei_foreground.zarr nuclei_binary.zarr -n 4 -v
scout nuclei morphology nuclei_binary.zarr centroids.npy nuclei_morphologies.csv -v
scout nuclei fluorescence centroids.npy nuclei_fluorescence sox2.zarr/ tbr1.zarr/ -v
