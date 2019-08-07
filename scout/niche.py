"""
Niche Module
=============

This module performs neighborhood analysis.

These include the following subcommands:

    - radial : compute radial profiles of each cell-type
    - proximity : compute average proximity to each cell-type
    - sample : randomly subsample cells to allow faster clustering
    - combine : combine single-cell features from multiple organoids
    - cluster : cluster cells into niches based on proximity
    - classify : fit niche classifier to subset and apply to all cells
    - name : assign names to niches

"""

import subprocess
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
import joblib
from MulticoreTSNE import MulticoreTSNE as TSNE
from scout.utils import verbose_print, write_csv
import matplotlib.pyplot as plt
import matplotlib.colors as cm


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

    # # Convert distances to proximity by average distance
    ave_distances = np.asarray([distances.mean(axis=-1) for distances in distances_list]).T
    # proximities = np.asarray([1 / (1 + ave_dist / r) for (ave_dist, r) in zip(ave_distances.T, radius)]).T

    # Convert distances to proximity by product
    proximities = np.asarray([(1 / (1 + dist / r)).prod(axis=-1) for (dist, r) in zip(distances_list, radius)]).T
    # proximities = np.asarray([(np.exp(-dist / r)).prod(axis=-1) for (dist, r) in zip(distances_list, radius)]).T

    return proximities


# Sampling

def randomly_sample(n, *items, return_idx=False):
    idx = np.arange(len(items[0]))
    np.random.shuffle(idx)
    idx = idx[:n]
    if return_idx:
        return tuple(item[idx] for item in items), idx
    else:
        return tuple(item[idx] for item in items)


# Clustering

def colormap_to_colors(n, name='Set2'):  # Not actually using this yet
    cmap = cm.get_cmap(name)
    colors = [tuple(list(cmap(i))[:3]) for i in range(n)]
    return colors


# Define command-line functionality

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
    radial_parser.add_argument('centroids', help="Path to input nuclei centroids in micron numpy array")
    radial_parser.add_argument('celltypes', help="Path to input cell-type labels numpy array")
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

    # Show plot
    if args.plot:
        idx = np.arange(len(proximities))
        np.random.shuffle(idx)
        idx = idx[:100000]
        plt.plot(proximities[idx, 0], proximities[idx, 1], '.', alpha=0.01)
        plt.show()

    # Save the proximities
    np.save(args.output, proximities)
    verbose_print(args, f'Proximities saved to {args.output}')

    verbose_print(args, f'Calculating proximities done!')


def proximity_cli(subparsers):
    proximity_parser = subparsers.add_parser('proximity', help="Calculate cell-type proxiomities",
                                             description='Calculates spatial proximities to each cell-type')
    proximity_parser.add_argument('centroids', help="Path to input nuclei centroids in micron numpy array")
    proximity_parser.add_argument('celltypes', help="Path to input cell-type labels numpy array")
    proximity_parser.add_argument('output', help="Path to output proximity numpy array")
    proximity_parser.add_argument('-r', help="Reference radii in micron for each cell-type", type=float, nargs='+', default=None)
    proximity_parser.add_argument('-k', help="Number of neighbors in proximity", type=int, default=3)
    proximity_parser.add_argument('-p', '--plot', help="Flag to show plots", action='store_true')
    proximity_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def gate_main(args):
    verbose_print(args, f'Gating cells into niches based on {args.proximity}')
    proximities = np.load(args.proximity)

    # # Gate proximities
    # adjacent_ax0 = (proximities[:, 0] > args.t[0])
    # adjacent_ax1 = (proximities[:, 1] > args.t[1])
    # btm_left = np.logical_not(np.logical_or(adjacent_ax0, adjacent_ax1))
    # btm_right = np.logical_and(adjacent_ax0, np.logical_not(adjacent_ax1))
    # top_left = np.logical_and(adjacent_ax1, np.logical_not(adjacent_ax0))
    # top_right = np.logical_and(adjacent_ax0, adjacent_ax1)
    # onehot = np.asarray([btm_left, btm_right, top_left, top_right])
    # labels = np.argmax(onehot, axis=0)
    high_right = (proximities[:, 0] > args.high[0])
    high_top = (proximities[:, 1] > args.high[1])
    low_right = (proximities[:, 0] > args.low[0])
    low_top = (proximities[:, 1] > args.low[1])

    btm_left = np.logical_not(np.logical_or(low_right, low_top))
    btm_right = np.logical_and(high_right, np.logical_not(high_top))
    top_left = np.logical_and(high_top, np.logical_not(high_right))
    top_right = np.logical_and(high_right, high_top)
    mid_left = np.logical_and(np.logical_and(low_top, np.logical_not(high_top)), np.logical_not(low_right))
    mid_right = np.logical_and(np.logical_and(low_right, np.logical_not(high_right)), np.logical_not(low_top))
    mid_inter = np.logical_and(np.logical_and(low_right, np.logical_not(high_right)), np.logical_and(low_top, np.logical_not(high_top)))

    onehot = np.asarray([btm_left, btm_right, top_left, top_right, mid_left, mid_right, mid_inter])
    labels = np.argmax(onehot, axis=0)

    if args.plot:
        proximities_sample, labels_sample = randomly_sample(100000, proximities, labels)
        names = ['DNeg', 'SOX2', 'TBR1', 'DPos', 'Mid-Left', 'Mid-Right', 'Mid-Inter']
        # Show proximities
        for i in range(len(names)):
            idx = np.where(labels_sample == i)[0]
            if len(idx) == 0:
                continue
            plt.plot(proximities_sample[idx, 0], proximities_sample[idx, 1], '.', alpha=args.alpha, label=names[i])
        # plt.legend()
        plt.show()

    # Save the niche labels
    np.save(args.labels, labels)
    verbose_print(args, f'Labels saved to {args.labels}')


def gate_cli(subparsers):
    gate_parser = subparsers.add_parser('gate', help="Gate cells into distinct niches",
                                        description='Gate cells into distinct niches based on proximities')
    gate_parser.add_argument('proximity', help="Path to input proximity numpy array")
    gate_parser.add_argument('labels', help="Path to output niche labels numpy array")
    gate_parser.add_argument('--low', help='Low proximity threshold', nargs='+', type=float, required=True)
    gate_parser.add_argument('--high', help='High proximity threshold', nargs='+', type=float, required=True)
    gate_parser.add_argument('-p', '--plot', help="Flag to show plots", action='store_true')
    gate_parser.add_argument('-a', '--alpha', help="Flag to show plots", type=float, default=0.01)
    gate_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def sample_main(args):
    if isinstance(args.inputs, list):
        inputs = args.inputs
    else:
        inputs = [args.inputs]
    if isinstance(args.outputs, list):
        outputs = args.outputs
    else:
        outputs = [args.outputs]
    if len(inputs) != len(outputs):
        raise ValueError("Number of inputs and outputs must match")

    verbose_print(args, f'Taking {args.samples} random samples from {args.inputs}')

    np.random.seed(args.seed)
    verbose_print(args, f'Random seed set to {args.seed}')

    # Load arrays
    input_arrs = [np.load(path) for path in args.inputs]

    # Randomly sample
    sampled_data, idx = randomly_sample(args.samples, *input_arrs, return_idx=True)

    # Save sample
    for output, samples in zip(outputs, sampled_data):
        np.save(output, samples)
        verbose_print(args, f'Saved samples to {output}')
    np.save(args.index, idx)
    verbose_print(args, f'Saved sample index to {args.index}')

    verbose_print(args, f'Randomly sampling done!')


def sample_cli(subparsers):
    sample_parser = subparsers.add_parser('sample', help="Randomly sample cells",
                                          description='Randomly sample cells before clustering')
    sample_parser.add_argument('samples', help="Number of samples to take", type=int)
    sample_parser.add_argument('index', help="Path to save sample index numpy array")
    sample_parser.add_argument('-i', '--inputs', help="Path to input numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-o', '--outputs', help="Path to sampled output numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-s', '--seed', help="Random seed", type=int, default=1)
    sample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def combine_main(args):
    verbose_print(args, f'Combining features from {len(args.inputs)} arrays')

    input_arrays = [np.load(path) for path in args.inputs]
    combined = np.concatenate(input_arrays, axis=args.a)

    verbose_print(args, f'Saving combined features to {args.output} with shape {combined.shape}')
    np.save(args.output, combined)

    verbose_print(args, f'Saving organoid labels to {args.sample}')
    names = np.concatenate([i*np.ones(len(arr)) for i, arr in enumerate(input_arrays)])
    np.save(args.sample, names)

    verbose_print(args, f'Combining features done!')


def combine_cli(subparsers):
    combine_parser = subparsers.add_parser('combine', help="Combine data from multiple organoids",
                                           description='Combine data from multiple organoids by concatenation')
    combine_parser.add_argument('inputs', help="Path to input numpy arrays", nargs='+')
    combine_parser.add_argument('-o', '--output', help="Path to output combined numpy array", required=True)
    combine_parser.add_argument('-s', '--sample', help="Path to output with sample name", required=True)
    combine_parser.add_argument('-a', help="Axis to concatenate", type=int, default=0)
    combine_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def tsne_main(args):
    verbose_print(args, f'Loaded niche labels from {args.labels}')
    labels = np.load(args.labels)

    verbose_print(args, f'Running t-SNE based on {args.proximity}')
    proximities = np.load(args.proximity)

    x_tsne = TSNE(n_components=2, n_jobs=-1, perplexity=800, learning_rate=100).fit_transform(proximities)

    if args.plot:
        # Show tSNE
        for i in range(4):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                continue
            plt.plot(x_tsne[idx, 0], x_tsne[idx, 1], '.', label=f'Cluster {i}')
        plt.legend()
        plt.show()

    # Save the t-SNE coordinates
    np.save(args.tsne, x_tsne)
    verbose_print(args, f't-SNE coordinates saved to {args.tsne}')

    verbose_print(args, f'Niche clustering done!')


def tsne_cli(subparsers):
    tsne_parser = subparsers.add_parser('cluster', help="Cluster cells into niches",
                                        description='Clusters cells into niches based on proximity to cell-types')
    tsne_parser.add_argument('proximity', help="Path to input proximity numpy array")
    tsne_parser.add_argument('labels', help="Path to input niche labels numpy array")
    tsne_parser.add_argument('tsne', help="Path to output t-SNE coordinates numpy array")
    tsne_parser.add_argument('-p', '--plot', help="Flag to show plots", action='store_true')
    tsne_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


# def classify_main(args):
#     verbose_print(args, f'Training logistic model based on {args.proximity_train} and {args.labels_train}')
#
#     # Load training data
#     x = np.load(args.proximity_train)
#     y = np.load(args.labels_train)
#     classes = np.unique(y)
#
#     if args.load is None:
#         verbose_print(args, f'Training new model')
#         # Train model
#         clf = LogisticRegression(random_state=0,
#                                  solver='lbfgs',
#                                  multi_class='multinomial',
#                                  n_jobs=-1).fit(x, y)
#         verbose_print(args, f'Training accuracy: {clf.score(x, y):.4f}')
#         verbose_print(args, f'Model coefficients:\n{clf.coef_}')
#         verbose_print(args, f'Model intercepts:\n{clf.intercept_}')
#     else:
#         verbose_print(args, f'Loading model from {args.load}')
#         clf = joblib.load(args.load)
#
#     if args.save is not None:
#         verbose_print(args, f'Saving model to {args.save}')
#         joblib.dump(clf, args.save)
#
#     # Apply classifier
#     proximities = np.load(args.proximity)
#     labels = clf.predict(proximities)
#
#     nb_cells = len(proximities)
#     verbose_print(args, f'Classified {nb_cells} cells into {len(classes)} niche classes')
#     for c in classes:
#         count = len(np.where(labels == c)[0])
#         verbose_print(args, f'Class {c}: {count:10d} cells {100 * count / nb_cells:10.3f}%')
#
#     verbose_print(args, f'Classifying done!')
#
#
# def classify_cli(subparsers):
#     classify_parser = subparsers.add_parser('classify', help="Cluster cells into niches",
#                                             description='Clusters cells into niches based on proximity to cell-types')
#     classify_parser.add_argument('proximity_train', help="Path to input proximity numpy array for training")
#     classify_parser.add_argument('labels_train', help="Path to output niche labels numpy array for training")
#     classify_parser.add_argument('proximity', help="Path to input proximity numpy array to classify")
#     classify_parser.add_argument('labels', help="Path to output niche labels numpy array")
#     classify_parser.add_argument('--save', help="Path to save trained model", default=None)
#     classify_parser.add_argument('--load', help="Path to load a trained model", default=None)
#     classify_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def name_main(args):
    verbose_print(args, f'Writing cluster names to {args.output}')
    write_csv(args.output, args.names)
    verbose_print(args, f'Naming done!')


def name_cli(subparsers):
    name_parser = subparsers.add_parser('name', help="Assign names to each group",
                                        description='Assign names to each group by writing to file')
    name_parser.add_argument('names', help="Names of each group", nargs='+')
    name_parser.add_argument('--output', '-o', help="Path to output names file", required=True)
    name_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def niche_main(args):
    commands_dict = {
        'radial': radial_main,
        'proximity': proximity_main,
        'gate': gate_main,
        'sample': sample_main,
        'combine': combine_main,
        'tsne': tsne_main,
        # 'classify': classify_main,
        'name': name_main,
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
    radial_cli(niche_subparsers)
    proximity_cli(niche_subparsers)
    gate_cli(niche_subparsers)
    sample_cli(niche_subparsers)
    combine_cli(niche_subparsers)
    tsne_cli(niche_subparsers)
    # classify_cli(niche_subparsers)
    name_cli(niche_subparsers)
    return niche_parser
