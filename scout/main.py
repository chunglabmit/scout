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

"""
Commands run on test data
--------------------------

scout preprocess tests/example.tif tests/example.zarr -s 200 -v

scout nuclei detect tests/data/syto.zarr tests/data/probability.zarr tests/data/centroids.npy -v
scout nuclei segment tests/data/probability.zarr tests/data/centroids.npy tests/data/foreground.zarr tests/data/binary.zarr tests/data/nuclei_segmentation.tif -v
scout nuclei fluorescence tests/data/centroids.npy tests/data/fluorescence_mfi.npy tests/data/fluorescence_stdev.npy tests/data/tbr1.zarr tests/data/sox2.zarr -g 0.7 1.0 1.0 -w 1 -v
scout nuclei gate tests/data/fluorescence_mfi.npy tests/data/gate_labels.npy 450 1150 -v -p
scout nuclei morphology tests/data/nuclei_segmentation.tif tests/data/centroids.npy tests/data/nuclei_morphology.csv -v

scout niche radial tests/data/centroids_um.npy tests/data/gate_labels.npy tests/data/niche_profiles.npy -v
scout niche proximity tests/data/centroids_um.npy tests/data/gate_labels.npy tests/data/niche_proximity.npy -v
scout niche sample 1000 -i tests/data/niche_proximity.npy -o tests/data/niche_proximity_sample.npy -v
scout niche cluster tests/data/niche_proximity_sample.npy tests/data/niche_labels_sample.npy tests/data/niche_tsne_sample.npy -v -n 4 -p
scout niche classify tests/data/niche_proximity_sample.npy tests/data/niche_labels_sample.npy tests/data/niche_proximity.npy tests/data/niche_labels.npy -v


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


Cyto
-----

Multiscale
-----------

Permute
--------

"""
