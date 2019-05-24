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
- scout permute (run permutation tests for features from multiple datasets -> save analysis results)
- scout jupyter (run the scout pipeline using interactive Jupyter notebooks)

"""

import argparse
import subprocess
import os

# Patch matplotlib backend for use on leviathan
import matplotlib; matplotlib.use('agg')

from scout.preprocess import preprocess_cli, preprocess_main
from scout.nuclei import nuclei_cli, nuclei_main
from scout.niche import niche_cli, niche_main
from scout.segment import segment_cli, segment_main
from scout.cyto import cyto_cli, cyto_main
from scout.multiscale import multiscale_cli, multiscale_main
from scout.permute import permute_cli, permute_main


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
    permute_parser = permute_cli(subparsers)
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
        'permute': permute_main,
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
scout preprocess histogram Ex_0_Em_0_destriped_stitched_master/ Ex0_hist.csv -s 50 -v
scout preprocess rescale Ex_0_Em_0_destriped_stitched_master/ Ex0_hist.csv Ex0_rescaled -t 120 -v (or 100)
scout preprocess denoise Ex0_rescaled/ Ex0_denoised -s 0.005 -v (or 0.001 for SOX2 TBR1)
scout preprocess convert Ex0_denoised/ syto.zarr -v

scout nuclei detect syto.zarr nuclei_probability.zarr centroids.npy --voxel-size voxel_size.csv --output-um centroids_um.npy -v
scout nuclei segment nuclei_probability.zarr centroids.npy nuclei_foreground.zarr nuclei_binary.zarr -v
scout nuclei fluorescence centroids.npy fluorescence sox2.zarr tbr1.zarr -g 0.7 1.0 1.0 -v
scout nuclei gate fluorescence/nuclei_mfis.npy fluorescence/gate_labels.npy 0.08 0.3 -p -v
scout nuclei morphology nuclei_binary.zarr centroids.npy nuclei_segmentations.npz nuclei_morphologies.csv -v

# scout niche radial tests/data/centroids_um.npy tests/data/gate_labels.npy tests/data/niche_profiles.npy -v
scout niche proximity data/centroids_um.npy data/gate_labels.npy data/niche_proximities.npy -r 5 5 -v
scout niche sample 10000 niche_sample_index.npy -i data/niche_proximities.npy -o data/niche_proximities_sample.npy -v
scout niche cluster data/niche_proximities_sample.npy data/niche_labels_sample.npy data/niche_tsne_sample.npy -v -n 4 -p
scout niche classify data/niche_proximities_sample.npy data/niche_labels_sample.npy data/niche_proximities.npy data/niche_labels.npy -v

scout segment downsample data/syto.zarr data/syto_down4x.tif 1 4 4 -v
scout segment ventricle data/syto_down4x.tif models/syto_vz_unet_200.pt data/segment_ventricles.tif -v
scout segment foreground data/syto_down4x.tif data/segment_foreground.tif -v -t 0.02 -g 8 4 4

scout cyto mesh data/segment_ventricles_exclude.tif data/voxel_size.csv data/mesh_ventricles.pkl -d 1 4 4 -g 2 -p -v
scout cyto profiles data/mesh_ventricles.pkl data/centroids_um.npy data/gate_labels.npy data/cyto_profiles.npy -v -p
scout cyto sample 5000 -i data/cyto_profiles.npy -o data/cyto_profiles_sample.npy -v
scout cyto cluster data/cyto_profiles_sample.npy data/cyto_labels_sample.npy data/cyto_tsne_sample.npy -n 8 -v
scout cyto classify data/cyto_profiles_sample.npy data/cyto_labels_sample.npy data/cyto_profiles.npy data/cyto_labels.npy -v



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
