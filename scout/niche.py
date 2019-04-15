"""
Niche Module
=============

This module performs neighborhood analysis.

These include the following subcommands:
    - radial : compute radial profiles of each cell-type
    - proximity : compute average proximity to each cell-type

"""

import subprocess
import numpy as np
from tqdm import tqdm
from joblib import dump, load
from sklearn.neighbors import NearestNeighbors
from scout.utils import verbose_print


# Query neighbors

def fit_neighbors(pts):
    nbrs = NearestNeighbors(algorithm='kd_tree', n_jobs=-1).fit(pts)
    return nbrs


def query_neighbors(nbrs, pts, n_neighbors):
    distances, indices = nbrs.kneighbors(pts, n_neighbors=n_neighbors)
    return distances, indices


def query_radius(nbrs, pts, radius):
    distances, indices = nbrs.radius_neighbors(pts, radius=radius)
    return distances, indices


# Calculate radial profiles

def make_bins(start, stop, bins):
    bin_edges = np.linspace(start, stop, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    return bin_edges, bin_width


def radial_profile(pts, distances, indices, radius, bins, labels=None):
    profiles = np.zeros((pts.shape[0], bins))

    vbin_edges, vbin_width = make_bins(0, 4/3*(radius + 1e-6)**3, bins)  # Equal-volume concentric shells
    bin_edges = (3/4*vbin_edges)**(1/3)

    for n, (pt, idx, dist) in tqdm(enumerate(zip(pts, indices, distances)), total=len(pts)):

        # Filter neighborhood by labels
        if labels is not None:
            neighborhood_labels = labels[idx]
            positive_idx = np.where(neighborhood_labels > 0)[0]
            idx = idx[positive_idx]
            dist = dist[positive_idx]  # select distances of the labelled cells

        # Skip if no cells
        if len(idx) == 0:
            continue

        # Calculate cell profile
        bin_idx = np.digitize(dist, bin_edges) - 1
        profiles[n] = np.bincount(bin_idx, minlength=bins)

    return profiles


# Calculate proximities


def proximity(centroids, celltypes, k, radius):
    # Find build k-d tree for each cell-type
    centroids_list = [centroids[np.where(labels)] for labels in celltypes.T]
    nbrs_list = [fit_neighbors(pts) for pts in centroids_list]

    # Find k-neighbors for each cell-type
    distances_list = []
    indices_list = []
    for i, nbrs in enumerate(nbrs_list):

        distances, indices = query_neighbors(nbrs, centroids, k)
        distances_list.append(distances)
        indices_list.append(indices)

    # Convert distances to proximity
    ave_distances = np.asarray([distances.mean(axis=-1) for distances in distances_list]).T
    proximities = np.asarray([1 / (1 + ave_dist / r) for (ave_dist, r) in zip(ave_distances.T, radius)]).T

    return proximities


# Define command-line functionality


# def fit_main(args):
#     verbose_print(args, f'Building k-d tree for {args.centroids}')
#
#     centroids = np.load(args.centroids)
#     nbrs = fit_neighbors(centroids)
#     dump(nbrs, args.neighbors, compress=args.c)
#
#     verbose_print(args, f'k-d tree saved to {args.output}')
#
#
# def fit_cli(subparsers):
#     fit_parser = subparsers.add_parser('fit', help="Fit k-d tree to nuclei centroids",
#                                        description='Builds k-d tree of nuclei centroids for neighbors calculations')
#     fit_parser.add_argument('centroids', help="Path to input nuclei centroids numpy array")
#     fit_parser.add_argument('neighbors', help="Path to output pickled k-d tree object")
#     fit_parser.add_argument('-c', help="Compression level for saving model", type=int, default=3)
#     fit_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def radial_main(args):
    verbose_print(args, f'Calculating radial profiles for {args.centroids}')

    # Load centroids and cell-type labels
    centroids = np.load(args.centroids)
    celltypes = np.load(args.celltypes)

    # May want to add subsampling here...

    # Find neighbors within a given radius
    nbrs = fit_neighbors(centroids)
    distances, indices = query_radius(nbrs, centroids, args.r)

    # Compute profiles for each cell-type
    profiles = np.zeros((celltypes.shape[-1], celltypes.shape[0], args.b))
    for i, labels in enumerate(celltypes.T):
        verbose_print(args, f'Counting cell-type {i}')
        profiles[i] = radial_profile(centroids, distances, indices, args.r, args.b, labels)

    # Save results
    np.save(args.output, profiles)
    verbose_print(args, f'Radial profiles saved to {args.output}')

    verbose_print(args, f'Calculating radial profiles done!')


def radial_cli(subparsers):
    radial_parser = subparsers.add_parser('radial', help="Calculate radial profiles",
                                          description='Calculates radial profiles for each cell-type')
    radial_parser.add_argument('centroids', help="Path to input nuclei centroids numpy array")
    radial_parser.add_argument('celltypes', help="Path to input cell-type labels numpy array")
    radial_parser.add_argument('neighbors', help="Path to input pickled k-d tree object")
    radial_parser.add_argument('output', help="Path to output profiles numpy array")
    radial_parser.add_argument('-r', help="Neighborhood radius", type=float, default=50)
    radial_parser.add_argument('-b', help="Number of bins in profile", type=float, default=5)
    radial_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def proximity_main(args):
    verbose_print(args, f'Calculating proximity to each cell-type for {args.centroids}')

    # Load centroids and cell-type labels
    centroids = np.load(args.centroids)
    celltypes = np.load(args.celltypes)

    # Check for any mismatch
    if args.r is None:
        radius = np.ones(celltypes.shape[-1])
        verbose_print(args, f'No reference radii specified... just using ones')
    else:
        radius = tuple(args.r)
        verbose_print(args, f'Using {radius} reference radii')
        if len(radius) != celltypes.shape[-1]:
            raise ValueError('The number of reference radii must match the number of provided cell-types')

    # May want to add subsampling here...

    # Calculate proximity to each cell-type
    proximities = proximity(centroids, celltypes, args.k, radius)

    # Save the proximities
    np.save(args.output, proximities)
    verbose_print(args, f'Proximities saved to {args.output}')

    verbose_print(args, f'Calculating proximities done!')


def proximity_cli(subparsers):
    proximity_parser = subparsers.add_parser('proximity', help="Calculate cell-type proxiomities",
                                             description='Calculates proximities for each cell-type')
    proximity_parser.add_argument('centroids', help="Path to input nuclei centroids numpy array")
    proximity_parser.add_argument('celltypes', help="Path to input cell-type labels numpy array")
    proximity_parser.add_argument('output', help="Path to output proximity numpy array")
    proximity_parser.add_argument('-r', help="Reference radii for each cell-type", type=float, nargs='+', default=None)
    proximity_parser.add_argument('-k', help="Number of neighbors in proximity", type=float, default=10)
    proximity_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


import matplotlib.pyplot as plt


def cluster_main(args):
    verbose_print(args, f'Clustering cells into niches based on {args.proximity}')

    proximities = np.load(args.proximity)

    plt.hist2d(proximities[:, 0], proximities[:, 1], bins=128)
    plt.show()



def cluster_cli(subparsers):
    cluster_parser = subparsers.add_parser('cluster', help="Cluster cells into niches",
                                           description='Clusters cells into niches based on proximity to cell-types')
    cluster_parser.add_argument('proximity', help="Path to input proximity numpy array")
    cluster_parser.add_argument('output', help="Path to output cluster labels numpy array")
    cluster_parser.add_argument('-n', help="Number of clusters", type=int, default=4)
    cluster_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def niche_main(args):
    commands_dict = {
        # 'fit': fit_main,
        'radial': radial_main,
        'proximity': proximity_main,
        'cluster': cluster_main,
    }
    func = commands_dict.get(args.niche_command, None)
    if func is None:
        print("Pickle Rick uses niche subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'niche', '-h'])
    else:
        func(args)


def niche_cli(subparsers):
    niche_parser = subparsers.add_parser('niche', help="niche analysis",
                                         description='Organoid niche analysis tool')
    niche_subparsers = niche_parser.add_subparsers(dest='niche_command', title='niche subcommands')
    # fit_cli(niche_subparsers)
    radial_cli(niche_subparsers)
    proximity_cli(niche_subparsers)
    cluster_cli(niche_subparsers)
    return niche_parser


"""

NICHE
------
cluster-niches
    centroids + cell-type labels -> sample -> proximities + densities -> tSNE + subset niche labels -> cluster stats
classify-niches
    proximities + subset niche labels -> train logistic model -> model weights + all niche labels
    
"""
