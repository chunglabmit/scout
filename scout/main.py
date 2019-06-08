"""
Main Module
============

This module exposes the command-line interfaces (CLIs) for running the SCOUT pipeline

The following CLIs are defined:

- scout preprocess (denoising, background removal, contrast, covnert)
- scout nuclei (nuclei prob, detection, cytometry, watershed, morphology -> save features)
- scout niche (neighborhoods, clustering, classify? -> save features)
- scout segment (ventricle, rasterize region labels, whole org -> save features)
- scout cyto (normals, profiles, clustering, classify -> save features)
- scout multiscale (build multiscale feature vectors from results)
- scout stats (run permutation tests for features from multiple datasets -> save analysis results)
- scout jupyter (run the scout pipeline using interactive Jupyter notebooks)

"""

import argparse
import subprocess
import os
from scout.preprocess import preprocess_cli, preprocess_main
from scout.nuclei import nuclei_cli, nuclei_main
from scout.niche import niche_cli, niche_main
from scout.segment import segment_cli, segment_main
from scout.cyto import cyto_cli, cyto_main
from scout.multiscale import multiscale_cli, multiscale_main
from scout.stats import stats_cli, stats_main


def jupyter_main(args):
    print(f'Starting Jupyter notebook on port {args.port}')
    scout_path = os.path.dirname(os.path.realpath(__file__))
    notebooks_path = os.path.join(scout_path, os.pardir, 'notebooks')
    subprocess.call(['jupyter', 'notebook', '--notebook-dir', notebooks_path, '--port', str(args.port)])


def jupyter_cli(subparsers):
    jupyter_parser = subparsers.add_parser('jupyter', help="launch pipeline as jupyter notebook",
                                           description="Jupyter notebook version of the analysis pipeline")
    jupyter_parser.add_argument('--port', '-p', help="Jupyter notebook port", default=8888, type=int)
    return jupyter_parser


def add_subparsers(parser):
    subparsers = parser.add_subparsers(dest='command', title='SCOUT subcommands')

    # Attach CLI subparsers to main scout parser
    preprocess_parser = preprocess_cli(subparsers)
    nuclei_parser = nuclei_cli(subparsers)
    niche_parser = niche_cli(subparsers)
    segment_parser = segment_cli(subparsers)
    cyto_parser = cyto_cli(subparsers)
    multiscale_parser = multiscale_cli(subparsers)
    stats_parser = stats_cli(subparsers)
    jupyter_parser = jupyter_cli(subparsers)


def main():
    parser = argparse.ArgumentParser(description="Multiscale organoid analysis tool")
    add_subparsers(parser)
    args = parser.parse_args()

    commands_dict = {
        'preprocess': preprocess_main,
        'nuclei': nuclei_main,
        'niche': niche_main,
        'segment': segment_main,
        'cyto': cyto_main,
        'multiscale': multiscale_main,
        'stats': stats_main,
        'jupyter': jupyter_main
    }

    func = commands_dict.get(args.command, None)
    if func is None:
        print("Prepare to phenotype some organoids! Wuba-luba-dub-dub!\n")
        subprocess.call(['scout', '-h'])
    else:
        func(args)


if __name__ == '__main__':
    main()


"""

Commands run on test data
--------------------------

# Preprocessing used for confocal images
scout preprocess data/syto.tif data/syto.zarr -t 0.05 -s 0.05 -v -p 8
scout preprocess data/tbr1.tif data/tbr1.zarr -t 0.05 -s 0.05 -v -p 8
scout preprocess data/sox2.tif data/sox2.zarr -t 0.05 -s 0.05 -v -p 8

# Preprocessing used for SPIM
scout preprocess histogram Ex_0_Em_0_stitched/ Ex0_hist.csv -s 50 -v
scout preprocess rescale Ex_0_Em_0_stitched/ Ex0_hist.csv Ex0_rescaled -t 120 -v (or 100) [-p 99.7]
# scout preprocess denoise Ex0_rescaled/ Ex0_denoised -s 0.005 -v (or 0.001 for SOX2 TBR1)
scout preprocess convert Ex0_rescaled/ syto.zarr -v -n 4

scout nuclei detect syto.zarr nuclei_probability.zarr centroids.npy --voxel-size voxel_size.csv --output-um centroids_um.npy -v
scout nuclei segment nuclei_probability.zarr centroids.npy nuclei_foreground.zarr nuclei_binary.zarr -v
scout nuclei morphology nuclei_binary.zarr centroids.npy nuclei_morphologies.csv -v [--segmentations nuclei_segmentations.npz]
scout nuclei fluorescence centroids.npy fluorescence sox2.zarr/ tbr1.zarr/ -g 0.7 1.0 1.0 -v
scout nuclei gate fluorescence/nuclei_mfis.npy nuclei_gating.npy 0.08 0.3 -p -v
scout nuclei name sox2 tbr1 dn -o celltype_names.csv -v

scout niche proximity centroids_um.npy nuclei_gating.npy niche_proximities.npy -r 25 25 -v -p -k 2
scout niche gate niche_proximities.npy niche_labels.npy --low 0.35 0.30 --high 0.66 0.63 -p -v --alpha 0.002
scout niche name DN SOX2 TBR1 DP Intermediate -o niche_names.csv -v
# Can skip the tSNE-related stuff for niches
scout niche sample 5000 niche_sample_index.npy -i niche_proximities.npy niche_labels.npy -o niche_proximities_sample.npy niche_labels_sample.npy -v
scout niche combine arr1.npy arr2.npy arr3.npy -o arr_combined.npy -s arr_samples.npy -v
scout niche tsne niche_proximities_combined.npy niche_labels_combined.npy niche_tsne_combined.npy -v -p

scout segment downsample Ex0_rescaled/ syto_down6x 6 6 -v -t
scout segment ventricle syto_down6x.tif /home/jswaney/scout/models/syto_vz_unet_200.pt segment_ventricles.tif -v
scout segment foreground syto_down6x.tif segment_foreground.tif -v -t 0.02 -g 8 4 4

scout cyto mesh segment_ventricles.tif voxel_size.csv mesh_ventricles.pkl -d 1 6 6 -g 2 -s 2 -p -v
scout cyto profiles mesh_ventricles.pkl centroids_um.npy nuclei_gating.npy cyto_profiles.npy -v -p
scout cyto sample 5000 cyto_sample_index.npy -i cyto_profiles.npy -o cyto_profiles_sample.npy -v
scout cyto combine sample1/cyto_profiles_sample.npy sample2/cyto_profiles_sample.npy -o cyto_profiles_combined.npy -s cyto_profiles_combined_samples.npy -v
scout cyto cluster cyto_profiles_combined.npy cyto_labels_combined.npy cyto_tsne_combined.npy -n 8 -v -p
scout cyto classify cyto_profiles_combined.npy cyto_labels_copmbined.npy cyto_profiles.npy cyto_labels.npy -v
scout cyto name name1 name2 ... -o cyto_names.csv -v


Input
-----
TIFF Images

Preprocess
-----------
+ denoised Zarr

Nuclei
-------
+ centroids
+ centroids_um
+ segmentation
+ MFIs / StDevs
+ gate labels
+ morphologies

Niche
------
+ profiles
+ proximities
+ proximities sample
+ niche labels sample
+ niche labels

Segment
--------
+ downsampled nuclei Zarr
+ ventricle segmentation
+ foreground segmentation

Cyto
-----
+ mesh parameters (mesh)
+ normal profiles (profiles)
+ normal profiles sample (sample)
+ normal labels sample (cluster)
+ normal labels (classify)

Multiscale
-----------
+ combine sampled features from each organoid

Permute
--------

"""
