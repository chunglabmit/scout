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
from sklearn.neighbors import NearestNeighbors
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib.colors as cm
from MulticoreTSNE import MulticoreTSNE as TSNE
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

    # # Convert distances to proximity by average distance
    # ave_distances = np.asarray([distances.mean(axis=-1) for distances in distances_list]).T
    # proximities = np.asarray([1 / (1 + ave_dist / r) for (ave_dist, r) in zip(ave_distances.T, radius)]).T

    # Convert distances to proximity by product
    proximities = np.asarray([(1 / (1 + dist / r)).prod(axis=-1) for (dist, r) in zip(distances_list, radius)]).T

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
        plt.plot(proximities[:, 0], proximities[:, 1], '.', alpha=0.01)
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
    sampled_data = randomly_sample(args.samples, *input_arrs, return_idx=False)

    # Save sample
    for output, samples in zip(outputs, sampled_data):
        np.save(output, samples)
        verbose_print(args, f'Saved samples to {output}')

    verbose_print(args, f'Randomly sampling done!')


def sample_cli(subparsers):
    sample_parser = subparsers.add_parser('sample', help="Randomly sample cells",
                                          description='Randomly sample cells before clustering')
    sample_parser.add_argument('samples', help="Number of samples to take", type=int)
    sample_parser.add_argument('-i', '--inputs', help="Path to input numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-o', '--outputs', help="Path to sampled output numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-s', '--seed', help="Random seed", type=int, default=1)
    sample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cluster_main(args):
    verbose_print(args, f'Clustering cells into niches based on {args.proximity}')

    proximities = np.load(args.proximity)

    # # GMM clustering
    # gmm = GaussianMixture(n_components=args.n, n_init=args.i).fit(proximities)
    #
    # verbose_print(args, f'GMM means:\n{gmm.means_}')
    # verbose_print(args, f'GMM weights (fractions):\n{gmm.weights_}')
    # verbose_print(args, f'GMM converged flag: {gmm.converged_}')
    # verbose_print(args, f'GMM ELBO: {gmm.lower_bound_:.6f}')
    # verbose_print(args, f'GMM BIC: {gmm.bic(proximities):.3f}')
    #
    # labels = gmm.predict(proximities)

    # K-means
    kmeans = KMeans(n_clusters=args.n, n_init=args.i).fit(proximities)

    verbose_print(args, f'Cluster centers:\n{kmeans.cluster_centers_}')
    verbose_print(args, f'Total inertia:\n{kmeans.inertia_:.8f}')

    labels = kmeans.predict(proximities)

    # # DBSCAN
    # dbscan = DBSCAN(eps=0.1, min_samples=2).fit(proximities)
    # labels = dbscan.labels_

    x_tsne = TSNE(n_components=2, n_jobs=-1, perplexity=800, learning_rate=100).fit_transform(proximities)

    if args.plot:
        # Show proximities
        for i in range(args.n):
            idx = np.where(labels == i)[0]
            plt.plot(proximities[idx, 0], proximities[idx, 1], '.', label=f'Cluster {i}')
        plt.legend()
        plt.show()
        # Show tSNE
        for i in range(args.n):
            idx = np.where(labels == i)[0]
            plt.plot(x_tsne[idx, 0], x_tsne[idx, 1], '.', label=f'Cluster {i}')
        plt.legend()
        plt.show()

    # Save the cluster labels and t-SNE coordinates
    np.save(args.labels, labels)
    verbose_print(args, f'Niche labels saved to {args.labels}')
    np.save(args.tsne, x_tsne)
    verbose_print(args, f't-SNE coordinates saved to {args.tsne}')

    verbose_print(args, f'Niche clustering done!')


def cluster_cli(subparsers):
    cluster_parser = subparsers.add_parser('cluster', help="Cluster cells into niches",
                                           description='Clusters cells into niches based on proximity to cell-types')
    cluster_parser.add_argument('proximity', help="Path to input proximity numpy array")
    cluster_parser.add_argument('labels', help="Path to output niche labels numpy array")
    cluster_parser.add_argument('tsne', help="Path to output t-SNE coordinates numpy array")
    cluster_parser.add_argument('-n', help="Number of clusters", type=int, default=4)
    cluster_parser.add_argument('-i', help="Number of KMeans initializations", type=int, default=10)
    cluster_parser.add_argument('-p', '--plot', help="Flag to show plots", action='store_true')
    cluster_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def classify_main(args):
    verbose_print(args, f'Training logistic model based on {args.proximity_train} and {args.labels_train}')

    # Load training data
    x = np.load(args.proximity_train)
    y = np.load(args.labels_train)

    classes = np.unique(y)

    # Train model
    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             multi_class='multinomial',
                             n_jobs=-1).fit(x, y)
    verbose_print(args, f'Training accuracy: {clf.score(x, y):.4f}')
    verbose_print(args, f'Model coefficients:\n{clf.coef_}')
    verbose_print(args, f'Model intercepts:\n{clf.intercept_}')

    # Apply classifier
    proximities = np.load(args.proximity)
    labels = clf.predict(proximities)

    nb_cells = len(proximities)
    verbose_print(args, f'Classified {nb_cells} cells into {len(classes)} niche classes')
    for c in classes:
        count = len(np.where(labels == c)[0])
        verbose_print(args, f'Class {c}: {count:10d} cells {100 * count / nb_cells:10.3f}%')

    # Save the niche labels
    np.save(args.labels, labels)

    verbose_print(args, f'Classifying done!')


def classify_cli(subparsers):
    classify_parser = subparsers.add_parser('classify', help="Cluster cells into niches",
                                            description='Clusters cells into niches based on proximity to cell-types')
    classify_parser.add_argument('proximity_train', help="Path to input proximity numpy array for training")
    classify_parser.add_argument('labels_train', help="Path to output niche labels numpy array for training")
    classify_parser.add_argument('proximity', help="Path to input proximity numpy array to classify")
    classify_parser.add_argument('labels', help="Path to output niche labels numpy array")
    classify_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def niche_main(args):
    commands_dict = {
        'radial': radial_main,
        'proximity': proximity_main,
        'sample': sample_main,
        'cluster': cluster_main,
        'classify': classify_main,
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
    sample_cli(niche_subparsers)
    cluster_cli(niche_subparsers)
    classify_cli(niche_subparsers)
    return niche_parser
