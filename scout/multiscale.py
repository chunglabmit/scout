"""
Multiscale Module
==================

This module builds the multiscale organoid feature vectors

These include the following subcommands:

    - features : consolidate multiscale features with optional metadata
    - combine : combine features from multiple organoids
    - summarize : display an organoid summary

"""

import os
import subprocess
import numpy as np
import pandas as pd
from skimage.measure import regionprops
from skimage.transform import resize
from skimage.morphology import binary_erosion
from scipy.ndimage import label
from sklearn.neighbors import NearestNeighbors
from scout import io
from scout.cyto import smooth_segmentation
from scout.utils import verbose_print, read_voxel_size, read_csv
import matplotlib.pyplot as plt


"""
Required files:

nuclei_gating.npy
nuclei_morphologies.csv
niche_proximities.npy
niche_labels.npy
celltype_names.csv
niche_names.csv
cyto_names.csv
cyto_profiles.npy
cyto_labels.npy
voxel_size.csv
segment_foreground.tif
segment_ventricles.tif
centroids_um.npy

Written output:

organoid_features.xlsx

"""


def singlecell_features(args, features, gate_labels, niche_labels, nuclei_morphologies, niche_proximities):
    # Get cell-type and niche names
    celltypes = read_csv(os.path.join(args.input, 'dataset/celltype_names.csv'))
    niches = read_csv(os.path.join(args.input, 'dataset/niche_names.csv'))

    # Niche cell counts
    print('cell counts')
    for n, niche_name in enumerate(niches):
        niche_idx = np.where(niche_labels == n)
        gate_labels_niche = gate_labels[niche_idx]
        niche_counts = gate_labels_niche.sum(axis=0)
        print("neighborhood", niche_name, "counts", niche_counts)
        for c, niche_count in enumerate(niche_counts):
            features[f'{niche_name} nbrhd, {celltypes[c]} count'] = niche_count

    # Niche proportions
    print('neighborhood proportions')
    niche_counts = []
    for n, niche_name in enumerate(niches):
        niche_idx = np.where(niche_labels == n)[0]
        niche_counts.append(len(niche_idx))
    niche_counts = np.asarray(niche_counts)
    total = niche_counts.sum()
    niche_fracs = niche_counts / total
    print(niche_fracs)
    for niche_name, niche_frac in zip(niches, niche_fracs):
        features[f'{niche_name} nbrhd fraction'] = niche_frac

    # Niche ratios
    print('neighborhood ratios')
    # TBR1 / SOX2 -> 2 / 1
    i, j = 2, 1
    ratio = niche_counts[i] / np.clip(niche_counts[j], 1, None)
    print(f'{niches[i]} / {niches[j]} ratio', ratio)
    features[f'{niches[i]} / {niches[j]} ratio'] = ratio
    # MidTBR1 / MidSOX2 -> 4 / 5
    i, j = 4, 5
    ratio = niche_counts[i] / np.clip(niche_counts[j], 1, None)
    print(f'{niches[i]} / {niches[j]} ratio', ratio)
    features[f'{niches[i]} / {niches[j]} ratio'] = ratio
    # MidInter / DN -> 6 / 0 (
    i, j = 6, 0
    ratio = niche_counts[i] / np.clip(niche_counts[j], 1, None)
    print(f'{niches[i]} / {niches[j]} ratio', ratio)
    features[f'{niches[i]} / {niches[j]} ratio'] = ratio

    # Niche equivalent diameters
    print('eq. diam')
    for n, niche_name in enumerate(niches):
        niche_idx = np.where(niche_labels == n)
        eq_diams = np.asarray(nuclei_morphologies['eq_diam'])
        eq_diams_niche = eq_diams[niche_idx]
        print(eq_diams_niche.mean(), eq_diams_niche.std())
        features[f'{niche_name} nbrhd, eq diam mean'] = eq_diams_niche.mean()
        features[f'{niche_name} nbrhd, eq diam stdev'] = eq_diams_niche.std()

    # Niche major lengths
    print('Major lengths')
    for n, niche_name in enumerate(niches):
        niche_idx = np.where(niche_labels == n)
        major_length = np.asarray(nuclei_morphologies['major_length'])
        major_length_niche = major_length[niche_idx]
        print(major_length_niche.mean(), major_length_niche.std())
        features[f'{niche_name} nbrhd, major axis mean'] = major_length_niche.mean()
        features[f'{niche_name} nbrhd, major axis stdev'] = major_length_niche.std()

    # Niche axis ratio
    print('axis ratio')
    for n, niche_name in enumerate(niches):
        niche_idx = np.where(niche_labels == n)
        axis_ratio = np.asarray(nuclei_morphologies['axis_ratio'])
        axis_ratio_niche = axis_ratio[niche_idx]
        print(axis_ratio_niche.mean(), axis_ratio_niche.std())
        features[f'{niche_name} nbrhd, axis ratio mean'] = axis_ratio_niche.mean()
        features[f'{niche_name} nbrhd, axis ratio stdev'] = axis_ratio_niche.std()

    # Niche proximities
    print('proximties')
    for n, niche_name in enumerate(niches):
        print('neighborhood', n)
        niche_idx = np.where(niche_labels == n)
        proximities = niche_proximities[niche_idx]
        gate_labels_niche = gate_labels[niche_idx]
        for c, celltype_name in enumerate(celltypes):
            print('cell type', c)
            celltype_idx = np.where(gate_labels_niche[:, c] == 1)[0]
            proximities_celltype = proximities[celltype_idx]
            print("SOX2, TBR1", proximities_celltype.mean(axis=0), proximities_celltype.std(axis=0), f"n = {len(celltype_idx)}")
            mean_proximities = proximities_celltype.mean(axis=0)
            stdev_proximities = proximities_celltype.std(axis=0)
            features[f'{niche_name} nbrhd, {celltype_name} proximity to SOX2 mean'] = mean_proximities[0]
            features[f'{niche_name} nbrhd, {celltype_name} proximity to SOX2 stdev'] = stdev_proximities[0]
            features[f'{niche_name} nbrhd, {celltype_name} proximity to TBR1 mean'] = mean_proximities[1]
            features[f'{niche_name} nbrhd, {celltype_name} proximity to TBR1 stdev'] = stdev_proximities[1]

    # 84 Single-cell so far
    return features


def cytoarchitecture_features(args, features):
    # Load the cytoarchitecture info needed
    celltypes = read_csv(os.path.join(args.input, 'dataset/celltype_names.csv'))
    cytoarchitectures = read_csv(os.path.join(args.input, 'cyto_names.csv'))
    cyto_profiles = np.load(os.path.join(args.input, 'dataset/cyto_profiles.npy'))
    cyto_labels = np.load(os.path.join(args.input, 'cyto_labels.npy'))
    print(cyto_profiles.shape, cyto_labels.shape)

    # Get cytoarchitecture freq
    # verbose_print(args, f'Loaded foreground segmentation: {foreground.shape}')
    for c, cyto_name in enumerate(cytoarchitectures):
        cyto_idx = np.where(cyto_labels == c)[0]
        cyto_count = len(cyto_idx)
        cyto_proportion = cyto_count / len(cyto_profiles)
        print(cyto_proportion)
        features[f'{cyto_name} cytoarchitecture fraction'] = cyto_proportion

    print('Average profile totals')
    # Get mean, stdev of cell counts in each profile for each cell type
    for c, cyto_name in enumerate(cytoarchitectures):
        cyto_idx = np.where(cyto_labels == c)[0]
        if len(cyto_idx) == 0:
            print(f'Cytoarchitecture {cyto_name} was empty')
            ave_counts = len(celltypes) * [np.nan]
        else:
            profiles = cyto_profiles[cyto_idx]
            ave_profiles = profiles.mean(axis=0)
            ave_counts = ave_profiles.sum(axis=-1)
        print(ave_counts)
        for i, celltype_name in enumerate(celltypes):
            features[f'ave. {cyto_name} profile, {celltype_name} count'] = ave_counts[i]

    print('Average profile means')
    # Get mean, stdev of each profile mean position for each cell type
    bin_positions = np.arange(cyto_profiles.shape[-1])
    for c, cyto_name in enumerate(cytoarchitectures):
        cyto_idx = np.where(cyto_labels == c)[0]
        if len(cyto_idx) == 0:
            print(f'Cytoarchitecture {cyto_name} was empty')
            ave_means = len(celltypes) * [np.nan]
        else:
            profiles = cyto_profiles[cyto_idx]
            ave_profiles = profiles.mean(axis=0)
            ave_means = (bin_positions * ave_profiles).sum(axis=-1) / ave_profiles.sum(-1)
        print(ave_means)
        for i, celltype_name in enumerate(celltypes):
            features[f'ave. {cyto_name} profile, {celltype_name} mean position'] = ave_means[i]

    print('Average profile stdev')
    # Get mean, stdev of each profile variance for each cell type
    bin_positions = np.arange(cyto_profiles.shape[-1])
    for c, cyto_name in enumerate(cytoarchitectures):
        cyto_idx = np.where(cyto_labels == c)[0]
        if len(cyto_idx) == 0:
            print(f'Cytoarchitecture {cyto_name} was empty')
            ave_stdev = len(celltypes) * [np.nan]
        else:
            profiles = cyto_profiles[cyto_idx]
            ave_profiles = profiles.mean(axis=0)
            ave_means = (bin_positions * ave_profiles).sum(axis=-1) / ave_profiles.sum(-1)
            ave_moment = (bin_positions ** 2 * ave_profiles).sum(axis=-1) / ave_profiles.sum(-1)
            ave_variance = ave_moment - ave_means ** 2
            ave_stdev = np.sqrt(ave_variance)
        print(ave_stdev)
        for i, celltype_name in enumerate(celltypes):
            features[f'ave. {cyto_name} profile, {celltype_name} profile stdev'] = ave_stdev[i]

    # 80 cytoarchitecture so far (if 8 clusters)
    return features


def wholeorg_features(args, features, gate_labels, niche_labels):
    celltypes = read_csv(os.path.join(args.input, 'dataset/celltype_names.csv'))
    niches = read_csv(os.path.join(args.input, 'dataset/niche_names.csv'))

    if args.g is not None:
        if len(args.g) == 1:
            sigma = args.g[0]
        else:
            sigma = tuple(args.g)

    if args.d is None:
        downsample_factor = 1
    else:
        downsample_factor = np.asarray(args.d)

    voxel_orig = read_voxel_size(os.path.join(args.input, 'dataset/voxel_size.csv'))
    verbose_print(args, f'Original voxel size: {voxel_orig}')

    voxel_down = tuple(voxel_orig * downsample_factor)
    verbose_print(args, f'Downsampled voxel size: {voxel_down}')

    # Overall organoid
    foreground = io.imread(os.path.join(args.input, 'dataset/segment_foreground.tif'))
    verbose_print(args, f'Loaded foreground segmentation: {foreground.shape}')

    if not np.allclose(voxel_down, max(voxel_down)):
        voxel_isotropic = tuple(max(voxel_down) * np.ones(len(voxel_down)))
        verbose_print(args, f'Resampling foreground to isotropic: {voxel_isotropic}')
        factors = np.asarray(voxel_isotropic) / np.asarray(voxel_down)
        shape_isotropic = tuple([int(s / f) for s, f in zip(foreground.shape, factors)])
        foreground = resize(foreground, output_shape=shape_isotropic, order=0).astype(foreground.dtype)
        verbose_print(args, f'Resampled foreground segmentation: {foreground.shape}')
    else:
        voxel_isotropic = voxel_down

    regions = regionprops(foreground)

    # Find largest region
    vol = 0
    idx = None
    for i, region in enumerate(regions):
        if region.area > vol:
            idx = i
    largest = regions[idx]

    volume_pixels = largest.area
    eq_diam_pixels = largest.equivalent_diameter
    major_axis_pixels = largest.major_axis_length
    minor_axis_pixels = largest.minor_axis_length

    volume_mm3 = volume_pixels * (np.asarray(voxel_isotropic) / 1000).prod()
    eq_diam_mm = eq_diam_pixels * voxel_isotropic[0] / 1000
    major_axis_mm = major_axis_pixels * voxel_isotropic[0] / 1000
    minor_axis_mm = minor_axis_pixels * voxel_isotropic[0] / 1000
    axis_ratio = major_axis_pixels / minor_axis_pixels

    print(f'Organoid volume (mm3): {volume_mm3:.3f}')
    print(f'Organoid equivalent diameter (mm): {eq_diam_mm:.3f}')
    print(f'Organoid major axis length (mm): {major_axis_mm:.3f}')
    print(f'Organoid minor axis length (mm): {minor_axis_mm:.3f}')
    print(f'Organoid axis ratio: {axis_ratio:.3f}')

    features[f'organoid volume (mm3)'] = volume_mm3
    features[f'organoid equivalent diameter (mm)'] = eq_diam_mm
    features[f'organoid major axis (mm)'] = major_axis_mm
    features[f'organoid minor axis (mm)'] = minor_axis_mm
    features[f'organoid axis ratio'] = axis_ratio

    # Ventricles
    ventricles = io.imread(os.path.join(args.input, 'dataset/segment_ventricles.tif'))
    verbose_print(args, f'Loaded ventricle segmentation: {ventricles.shape}')

    # Smooth segmentation
    if args.g is not None:
        ventricles = smooth_segmentation(ventricles, sigma) > 0.5
        verbose_print(args, f'Smoothed segmentation with sigma {sigma}')

    if not np.allclose(voxel_down, max(voxel_down)):
        voxel_isotropic = tuple(max(voxel_down) * np.ones(len(voxel_down)))
        verbose_print(args, f'Resampling ventricles to isotropic: {voxel_isotropic}')
        factors = np.asarray(voxel_isotropic) / np.asarray(voxel_down)
        shape_isotropic = tuple([int(s / f) for s, f in zip(ventricles.shape, factors)])
        ventricles = resize(ventricles, output_shape=shape_isotropic, order=0, preserve_range=True).astype(
            ventricles.dtype)
        verbose_print(args, f'Resampled ventricle segmentation: {ventricles.shape}')
    else:
        voxel_isotropic = voxel_down

    labels, nb_ventricles = label(ventricles)
    verbose_print(args, f'Found {nb_ventricles} connected components in ventricle segmentation')

    regions = regionprops(labels)

    volumes_pixels = np.asarray([region.area for region in regions])
    eq_diams_pixels = np.asarray([region.equivalent_diameter for region in regions])
    major_axes_pixels = np.asarray([region.major_axis_length for region in regions])
    minor_axes_pixels = np.asarray([region.minor_axis_length for region in regions])

    volumes_um3 = volumes_pixels * np.asarray(voxel_isotropic).prod()
    eq_diams_um = eq_diams_pixels * voxel_isotropic[0]
    major_axes_um = major_axes_pixels * voxel_isotropic[0]
    minor_axes_um = minor_axes_pixels * voxel_isotropic[0]
    axis_ratios = major_axes_pixels / np.clip(minor_axes_pixels, 1, None)

    ave_volume_um3 = volumes_um3.mean()
    stdev_volume_um3 = volumes_um3.std()
    ave_eq_diam_um = eq_diams_um.mean()
    stdev_eq_diam_um = eq_diams_um.std()
    ave_major_axis_um = major_axes_um.mean()
    stdev_major_axis_um = major_axes_um.std()
    ave_minor_axis_um = minor_axes_um.mean()
    stdev_minor_axis_um = minor_axes_um.std()
    ave_axis_ratio = axis_ratios.mean()
    stdev_axis_ratio = axis_ratios.std()

    print(f'ave. ventricle volume (um3): {ave_volume_um3:.3f} ({stdev_volume_um3:.3f})')
    print(f'ave. ventricle equivalent diameter (um): {ave_eq_diam_um:.3f} ({stdev_eq_diam_um:.3f})')
    print(f'ave. ventricle major axis length (um): {ave_major_axis_um:.3f} ({stdev_major_axis_um:.3f})')
    print(f'ave. ventricle minor axis length (mm): {ave_minor_axis_um:.3f} ({stdev_minor_axis_um:.3f})')
    print(f'ave. ventricle axis ratio: {ave_axis_ratio:.3f} ({stdev_axis_ratio:.3f})')

    features[f'ventricle count'] = nb_ventricles
    features[f'ventricle volume mean (um3)'] = ave_volume_um3
    features[f'ventricle volume stdev (um3)'] = stdev_volume_um3
    features[f'ventricle equivalent diameter mean (um)'] = ave_eq_diam_um
    features[f'ventricle equivalent diameter stdev (um)'] = stdev_eq_diam_um
    features[f'ventricle major axis mean (um)'] = ave_major_axis_um
    features[f'ventricle major axis stdev (um)'] = stdev_major_axis_um
    features[f'ventricle minor axis mean (um)'] = ave_minor_axis_um
    features[f'ventricle minor axis stdev (um)'] = stdev_minor_axis_um
    features[f'ventricle axis ratio mean'] = ave_axis_ratio
    features[f'ventricle axis ratio stdev'] = stdev_axis_ratio

    # Distance to surface
    mask = foreground > 0
    verbose_print(args, f'Made foreground mask: {mask.shape}')

    # Find surface coordinates
    eroded = binary_erosion(mask)
    surface = np.logical_and(mask, np.logical_not(eroded))
    coords = np.asarray(np.where(surface)).T
    surface_points = coords * np.asarray(voxel_down)

    # Load cell centers
    centroids_um = np.load(os.path.join(args.input, 'dataset/centroids_um.npy'))

    # Query nearest surface point for each cell center
    print('Surface distances')
    nbrs = NearestNeighbors(n_neighbors=1).fit(surface_points)
    for n, niche_name in enumerate(niches):
        print('neighborhood', n)
        niche_idx = np.where(niche_labels == n)[0]  # This is an index into centoids_um
        niche_centroids_um = centroids_um[niche_idx]
        gate_labels_niche = gate_labels[niche_idx]
        for c, celltype_name in enumerate(celltypes):
            print('cell type', c)
            celltype_idx = np.where(gate_labels_niche[:, c] == 1)[0]
            if len(celltype_idx) > 0:
                niche_celltype_centroids_um = niche_centroids_um[celltype_idx]
                surface_dist, _ = nbrs.kneighbors(niche_celltype_centroids_um)
                ave_surface_dist = surface_dist.mean()
                stdev_surface_dist = surface_dist.std()
            else:
                ave_surface_dist = np.nan
                stdev_surface_dist = np.nan
            print(f'Ave. surface dist: {ave_surface_dist:.3f} ({stdev_surface_dist:.3f}, n = {len(celltype_idx)})')
            features[f'{niche_name} nbrhd, {celltype_name} surface distance mean (um)'] = ave_surface_dist
            features[f'{niche_name} nbrhd, {celltype_name} surface distance stdev (um)'] = stdev_surface_dist

    return features


def features_main(args):
    verbose_print(args, f'Calculating multiscale features')

    # Identfy all datasets to be analyzed
    if os.path.isdir(args.input):
        input_folders = [os.path.basename(os.path.abspath(args.input))]
    elif os.path.splitext(os.path.abspath(args.input))[1] == '.csv':
        analysis = pd.read_csv(os.path.abspath(args.input), index_col=0)
        parent_dir = os.path.abspath(os.path.join(os.path.abspath(args.input), os.pardir))
        input_folders = [os.path.join(parent_dir, t, f) for t, f in zip(analysis['type'], analysis.index)]
    else:
        raise ValueError('Input must be a folder with a symlinked dataset or an analysis CSV file')

    # Analyze each dataset
    for input_folder in input_folders:
        verbose_print(args, f'Calculating multiscale features for {os.path.basename(input_folder)}')

        # inject current folder path into command line arguments
        args.input = os.path.abspath(input_folder)

        # Create a dictionary for holding all features
        features = {'dataset': os.path.basename(args.input)}

        # Load all single-cell data
        verbose_print(args, f'Loading input single cell measurements')
        gate_labels = np.load(os.path.join(args.input, 'dataset/nuclei_gating.npy'))
        nuclei_morphologies = pd.read_csv(os.path.join(args.input, 'dataset/nuclei_morphologies.csv'))
        niche_proximities = np.load(os.path.join(args.input, 'dataset/niche_proximities.npy'))
        niche_labels = np.load(os.path.join(args.input, 'dataset/niche_labels.npy'))

        # Add in double negatives
        # TODO: Move this to nuclei module
        negatives = np.logical_and(gate_labels[:, 0] == 0, gate_labels[:, 1] == 0)
        gate_labels = np.hstack([gate_labels, negatives[:, np.newaxis]])

        # Calculate multiscale features
        features = singlecell_features(args, features, gate_labels, niche_labels, nuclei_morphologies, niche_proximities)
        features = cytoarchitecture_features(args, features)
        features = wholeorg_features(args, features, gate_labels, niche_labels)

        # Save results
        df = pd.Series(features)
        df.to_excel(os.path.join(args.input, 'organoid_features.xlsx'))

    verbose_print(args, f'Multiscale features done!')


def features_cli(subparsers):
    features_parser = subparsers.add_parser('features', help="Compute multiscale features",
                                            description='Compute multiscale features for an organoid')
    features_parser.add_argument('input', help="Path to input organoid folder or analysis CSV file")
    features_parser.add_argument('-d', help="Downsampling factors from voxel size file", type=int, nargs='+', default=None)
    features_parser.add_argument('-g', help="Amount of gaussian smoothing", type=float, nargs='+', default=None)
    features_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def combine_main(args):
    verbose_print(args, f'Combining multiscale features')

    # Identfy all datasets to be analyzed if passed analysis CSV
    if os.path.splitext(args.inputs[0])[1] == '.csv':
        analysis = pd.read_csv(args.inputs[0], index_col=0)
        parent_dir = os.path.abspath(os.path.join(os.path.abspath(args.inputs[0]), os.pardir))
        args.inputs = [os.path.join(parent_dir, t, f) for t, f in zip(analysis['type'], analysis.index)]

    dfs = []
    for organoid in args.inputs:
        path = os.path.join(organoid, 'organoid_features.xlsx')
        dfs.append(pd.read_excel(path, index_col=0))
    print(dfs[0].head())
    df = pd.concat(dfs, axis=1, sort=False)
    df.to_excel(args.output)

    verbose_print(args, f'Combining multiscale features done!')


def combine_cli(subparsers):
    combine_parser = subparsers.add_parser('combine', help="Combine organoid features",
                                           description='Combine organoid features for a study')
    combine_parser.add_argument('inputs', help="Path to input organoid folders or analysis CSV file", nargs='+')
    combine_parser.add_argument('--output', '-o', help="Path to output Excel table", default='combined_features.xlsx')
    combine_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def setup_main(args):
    verbose_print(args, f'Setting up analysis folder')

    # Load the CSV as a dataframe
    df = pd.read_csv(args.input, index_col=0)

    # Create folders for each group
    groups = list(set(df['type']))
    groups.sort()
    for group in groups:
        verbose_print(args, f'Making directory for {group} group')
        os.makedirs(os.path.join(args.output, group), exist_ok=True)

    # Create folders for each dataset with symlinks to underlying data
    for path in df.index:
        group = df['type'].loc[path]
        new_dir = os.path.join(args.output, group, path)
        verbose_print(args, f'Making directory and symlink for {path}')
        os.makedirs(new_dir, exist_ok=True)
        os.symlink(os.path.join(os.path.abspath(args.datasets), path),
                   os.path.join(os.path.abspath(new_dir), 'dataset'))

    verbose_print(args, f'Done setting up analysis folder!')


def setup_cli(subparsers):
    setup_parser = subparsers.add_parser('setup', help="Setup analysis folder",
                                         description='Create folders with simlinks to datasets for comparison')
    setup_parser.add_argument('input', help="Path to input analysis CSV")
    setup_parser.add_argument('datasets', help="Path to folder containing datasets")
    setup_parser.add_argument('--output', help="Path to target analysis directory", default='.')
    setup_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def select_main(args):
    verbose_print(args, f'Selecting datasets for analysis')

    # Load dataset CSV and select datasets by group
    df = pd.read_csv(args.input, index_col=0)
    groups = [df.where(df['type'] == g).dropna() for g in args.groups]
    for g, name in zip(groups, args.groups):
        verbose_print(args, f'Found {len(g)} datasets in group {name}')

    # Create output CSV
    df2 = pd.concat(groups)
    df2.to_csv(args.output)

    verbose_print(args, f'Done selecting datasets!')


def select_cli(subparsers):
    select_parser = subparsers.add_parser('select', help="Select datasets for analysis",
                                          description='Create analysis CSV by selecting groups from dataset summary')
    select_parser.add_argument('input', help="Path to dataset summary CSV")
    select_parser.add_argument('groups', help="Names of dataset types to select", nargs='+')
    select_parser.add_argument('--output', help="Path to output CSV", default='analysis.csv')
    select_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def multiscale_cli(subparsers):
    multiscale_parser = subparsers.add_parser('multiscale', help="multiscale features",
                                              description="Build features for multiscale organoid analysis")
    multiscale_parser = multiscale_parser.add_subparsers(dest='multiscale_command', title='multiscale subcommands')
    features_cli(multiscale_parser)
    combine_cli(multiscale_parser)
    setup_cli(multiscale_parser)
    select_cli(multiscale_parser)
    return multiscale_parser


def multiscale_main(args):
    commands_dict = {
        'select': select_main,
        'setup': setup_main,
        'features': features_main,
        'combine': combine_main,
    }
    func = commands_dict.get(args.multiscale_command, None)
    if func is None:
        print("Pickle Rick uses multiscale subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'multiscale', '-h'])
    else:
        func(args)
