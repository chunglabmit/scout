"""
Niche Module
=============

This module performs cytometry and neighborhood analysis.

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


# Define command-line functionality


def fit_main(args):
    verbose_print(args, f'Building k-d tree for {args.centroids}')

    centroids = np.load(args.centroids)
    nbrs = fit_neighbors(centroids)
    dump(nbrs, args.neighbors, compress=args.c)

    verbose_print(args, f'k-d tree saved to {args.output}')


def fit_cli(subparsers):
    fit_parser = subparsers.add_parser('fit', help="Fit k-d tree to nuclei centroids",
                                       description='Builds k-d tree of nuclei centroids for neighbors calculations')
    fit_parser.add_argument('centroids', help="Path to input nuclei centroids numpy array")
    fit_parser.add_argument('neighbors', help="Path to output pickled k-d tree object")
    fit_parser.add_argument('-c', help="Compression level for saving model", type=int, default=3)
    fit_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def radial_main(args):
    verbose_print(args, f'Calculating radial profiles for {args.centroids}')

    # Load centroids and cell-type labels
    centroids = np.load(args.centroids)
    celltypes = np.load(args.celltypes)

    # May want to add subsampling here...

    # Find neighbors within a given radius
    nbrs = load(args.neighbors)
    distances, indices = query_radius(nbrs, centroids, args.r)

    # Compute profiles for each cell-type
    profiles = np.zeros((celltypes.shape[-1], celltypes.shape[0], args.b))
    for i, labels in enumerate(celltypes.T):
        verbose_print(args, f'Counting cell-type {i}...')
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


def niche_main(args):
    commands_dict = {
        'fit': fit_main,
        'radial': radial_main,
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
    fit_cli(niche_subparsers)
    radial_cli(niche_subparsers)
    return niche_parser


"""

NICHE
------
fit-neighbors
    centroids -> fit NearestNeighbors object
cluster-niches
    centroids + cell-type labels -> sample -> proximities + densities -> tSNE + subset niche labels -> cluster stats
classify-niches
    proximities + subset niche labels -> train logistic model -> model weights + all niche labels
    
"""
