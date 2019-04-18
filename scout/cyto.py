"""
Cyto Module
============

This module performs organoid cytoarchitecture analysis

These include the following subcommands:
    - mesh : compute surface mesh from segmentation
    - profiles : compute profiles along surface normals
    - sample : randomly sample profiles
    - cluster : cluster profiles into cytoarchitectures
    - classify : classify profiles into cytoacritectures

"""

import multiprocessing
import pickle
import subprocess
import numpy as np
from tqdm import tqdm
from skimage.measure import marching_cubes_lewiner
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from MulticoreTSNE import MulticoreTSNE as TSNE
from mayavi import mlab
import matplotlib.pyplot as plt
from scout.preprocess import gaussian_blur
from scout.niche import sample_main
from scout import io
from scout.utils import verbose_print, read_voxel_size


# Meshing and normals

def smooth_segmentation(seg, sigma=1, scale_factor=1):
    binary = (seg > 0)
    smooth = scale_factor * gaussian_blur(binary, sigma)
    return smooth.astype(np.float32)


def marching_cubes(seg, level, spacing, step_size):
    return marching_cubes_lewiner(seg, level=level, spacing=spacing, step_size=step_size, allow_degenerate=False)


def save_mesh(path, mesh):
    with open(path, 'wb') as f:
        pickle.dump(mesh, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_mesh(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


# Plotting

def plot_mesh(verts, faces, color=(1, 0, 0), figure=None):
    if figure is not None:
        mlab.figure(figure)
    return mlab.triangular_mesh([vert[0] for vert in verts],
                                [vert[1] for vert in verts],
                                [vert[2] for vert in verts],
                                faces,
                                color=color)


def randomly_sample(n, *items, return_idx=False):
    idx = np.arange(len(items[0]))
    np.random.shuffle(idx)
    idx = idx[:n]
    if return_idx:
        return tuple(item[idx] for item in items), idx
    else:
        return tuple(item[idx] for item in items)


def plot_nuclei(centers_um, nb_nuclei, sox2_labels, tbr1_labels, scale_factor=1, figure=None):
    if figure is not None:
        mlab.figure(figure)
    centers_sample, sox2_labels_sample, tbr1_labels_sample = randomly_sample(nb_nuclei,
                                                                             centers_um,
                                                                             sox2_labels,
                                                                             tbr1_labels)

    negative_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample == 0))[0]
    sox2_idx = np.where(np.logical_and(sox2_labels_sample > 0, tbr1_labels_sample == 0))[0]
    tbr1_idx = np.where(np.logical_and(sox2_labels_sample == 0, tbr1_labels_sample > 0))[0]

    negative = centers_sample[negative_idx]
    sox2 = centers_sample[sox2_idx]
    tbr1 = centers_sample[tbr1_idx]

    # Plot nuclei
    mlab.points3d(negative[:, 0], negative[:, 1], negative[:, 2], scale_factor=scale_factor, color=(0, 0, 1))
    mlab.points3d(sox2[:, 0], sox2[:, 1], sox2[:, 2], scale_factor=scale_factor, color=(1, 0, 0))
    mlab.points3d(tbr1[:, 0], tbr1[:, 1], tbr1[:, 2], scale_factor=scale_factor, color=(0, 1, 0))


# Cell-type profiles

def make_bins(start, stop, bins):
    bin_edges = np.linspace(start, stop, bins + 1)
    bin_width = bin_edges[1] - bin_edges[0]
    return bin_edges, bin_width


def cross_products(vectors, ref=np.array([1, 0, 0])):
    return np.cross(vectors, ref)


def dot_products(vectors, ref=np.array([1, 0, 0])):
    return np.dot(vectors, ref)


centers_um_global = None
sox2_labels_global = None
tbr1_labels_global = None


def compute_profile(vert, vi, ci, length, bins, radius):
    global centers_um_global
    global sox2_labels_global
    global tbr1_labels_global

    pts = centers_um_global
    sox2_labels = sox2_labels_global
    tbr1_labels = tbr1_labels_global

    # Translate points to origin
    pts_translated = pts - vert

    # Rotate points to align the normal with the z-axis
    v_cross = np.array([[0, -vi[2], vi[1]],
                        [vi[2], 0, -vi[0]],
                        [-vi[1], vi[0], 0]])
    rotation_matrix = np.eye(3) + v_cross + np.matmul(v_cross, v_cross) / (1 + ci)
    pts_translated_rotated = rotation_matrix.dot(pts_translated.T).T

    # Bin count the cells
    bin_edges, bin_height = make_bins(0, length, bins)
    sox2_count = np.zeros(bins, np.int)
    tbr1_count = np.zeros(bins, np.int)
    negative_count = np.zeros(bins, np.int)

    for j, bin_start in enumerate(bin_edges[:-1]):
        bin_stop = bin_start + bin_height
        x, y, z = pts_translated_rotated[:, 2], pts_translated_rotated[:, 1], pts_translated_rotated[:, 0]

        idx = np.where(np.logical_and(x ** 2 + y ** 2 <= radius ** 2, np.logical_and(z >= bin_start, z <= bin_stop)))[0]

        sox2_lbls = sox2_labels[idx]
        tbr1_lbls = tbr1_labels[idx]
        negative_lbls = np.where(np.logical_and(sox2_lbls == 0, tbr1_lbls == 0))[0]

        sox2_count[j] = sox2_lbls.sum()
        tbr1_count[j] = tbr1_lbls.sum()
        negative_count[j] = len(negative_lbls)

    return sox2_count, tbr1_count, negative_count


def _compute_profile(inputs):
    return compute_profile(*inputs)


def compute_profiles(verts, normals, length, bins, radius, centers_um, sox2_labels, tbr1_labels):
    global centers_um_global
    global sox2_labels_global
    global tbr1_labels_global

    centers_um_global = centers_um
    sox2_labels_global = sox2_labels
    tbr1_labels_global = tbr1_labels

    v = cross_products(normals)
    c = dot_products(normals)

    # Get cell density profiles for each cell-type
    args_list = []
    for i, (vi, ci, vert) in enumerate(zip(v, c, verts)):
        args_list.append((vert, vi, ci, length, bins, radius))

    with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
        results = list(tqdm(pool.imap(_compute_profile, args_list), total=len(args_list)))
    return np.asarray(results)


# Profile clustering

def profiles_to_features(profiles):
    features = profiles.reshape((len(profiles), -1)).astype(np.float)  # Flattened profiles
    return scale(features)  # Normalize each feature (cell bin) to unit mean, zero variance


# Define command-line functionality

# def smooth_main(args):
#     verbose_print(args, f'Smoothing segmentation at {args.input}')
#
#     # Load the segmentation
#     seg = io.imread(args.input)
#
#     # Smooth the segmentation to float
#     smoothed = smooth_segmentation(seg, args.g, args.f)
#
#     # Save the result
#     io.imsave(args.output, smoothed, compress=3)
#     verbose_print(args, f'Smoothed segmentation saved to {args.output}')
#
#     verbose_print(args, 'Smoothing done!')
#
#
# def smooth_cli(subparsers):
#     smooth_parser = subparsers.add_parser('smooth', help="Smooth a segmentation",
#                                           description='Smooth a binary segmentation to float')
#     smooth_parser.add_argument('input', help="Path to input segmentation TIFF")
#     smooth_parser.add_argument('output', help="Path to output smoothed segmentation TIFF")
#
#     smooth_parser.add_argument('-f', help="Scale factor for smoothed segmentation", type=float, default=1.0)
#     smooth_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def mesh_main(args):
    if args.g is not None:
        if len(args.g) == 1:
            sigma = args.g[0]
        else:
            sigma = tuple(args.g)

    if args.d is None:
        downsample_factor = 1
    else:
        downsample_factor = np.asarray(args.d)

    verbose_print(args, f'Meshing segmentation at {args.input}')

    # Calculate the downsampled voxel size
    voxel_orig = read_voxel_size(args.voxel_size)
    voxel_down = tuple(voxel_orig * downsample_factor)
    verbose_print(args, f'Original voxel size (um): {voxel_orig}')
    verbose_print(args, f'Downsampled voxel size (um): {voxel_down}')

    # Load segmentation
    seg = io.imread(args.input)

    # Smooth segmentation
    if args.g is not None:
        seg = smooth_segmentation(seg, sigma)
        verbose_print(args, f'Smoothed segmentation with sigma {sigma}')

    # Calculate mesh surface
    verts, faces, normals, values = marching_cubes(seg, args.l, voxel_down, args.s)
    mesh = {'verts': verts, 'faces': faces, 'normals': normals, 'values': values}

    # Plot mesh
    if args.plot:
        plot_mesh(mesh['verts'], mesh['faces'])
        mlab.show()

    # Save mesh
    save_mesh(args.output, mesh)
    verbose_print(args, f'Mesh saved to {args.output}')

    verbose_print(args, 'Meshing done!')


def mesh_cli(subparsers):
    mesh_parser = subparsers.add_parser('mesh', help="Mesh segmentation to surface",
                                        description='Mesh a segmentation to surface using marching cubes')
    mesh_parser.add_argument('input', help="Path to input segmentation TIFF")
    mesh_parser.add_argument('voxel_size', help="Path to original voxel size CSV")
    mesh_parser.add_argument('output', help="Path to output mesh")
    mesh_parser.add_argument('-d', help="Downsampling factors from voxel size file", type=int, nargs='+', default=None)
    mesh_parser.add_argument('-g', help="Amount of gaussian smoothing", type=float, nargs='+', default=None)
    mesh_parser.add_argument('-l', help='Isolevel for surface after smoothing', type=float, default=0.2)
    mesh_parser.add_argument('-s', help="Step size for mesh", type=int, default=1)
    mesh_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    mesh_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def profiles_main(args):
    verbose_print(args, f'Calculating profiles from {args.mesh}')

    # Get vertices and normals
    mesh = load_mesh(args.mesh)
    verts = mesh['verts']
    normals = mesh['normals']

    # Load centers and labels
    centroids_um = np.load(args.centroids)
    labels = np.load(args.labels)

    tbr1_labels = labels[:, 0]
    sox2_labels = labels[:, 1]

    # Plot mesh
    if args.plot:
        plot_mesh(mesh['verts'], mesh['faces'])
        plot_nuclei(centroids_um, 10000, sox2_labels, tbr1_labels, scale_factor=8)
        mlab.show()

    # Calculate profiles
    verbose_print(args, f'Progress:')
    profiles = compute_profiles(verts, normals, args.l, args.b, args.r, centroids_um, sox2_labels, tbr1_labels)

    # Save the profiles
    np.save(args.output, profiles)
    verbose_print(args, f'Profiles saved to {args.output}')

    verbose_print(args, 'Calculating profiles done!')


def profiles_cli(subparsers):
    profiles_parser = subparsers.add_parser('profiles', help="Compute profiles along normals",
                                            description='Compute cell-type profiles along ventricle normals')
    profiles_parser.add_argument('mesh', help="Path to input mesh")
    profiles_parser.add_argument('centroids', help="Path to input centroids in micron")
    profiles_parser.add_argument('labels', help="Path to cell-type labels")
    profiles_parser.add_argument('output', help="Path to output profiles numpy array")
    profiles_parser.add_argument('-l', help='Length in micron of the profiles', type=float, default=200)
    profiles_parser.add_argument('-r', help='Radius of profile bins', type=int, default=20)
    profiles_parser.add_argument('-b', help='Number of bins in each profile', type=int, default=5)
    profiles_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    profiles_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def sample_cli(subparsers):
    sample_parser = subparsers.add_parser('sample', help="Randomly sample profiles",
                                          description='Randomly sample profiles before clustering')
    sample_parser.add_argument('samples', help="Number of samples to take", type=int)
    sample_parser.add_argument('-i', '--inputs', help="Path to input numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-o', '--outputs', help="Path to sampled output numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-s', '--seed', help="Random seed", type=int, default=1)
    sample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cluster_main(args):
    verbose_print(args, f'Clustering profiles from {args.input}')

    # Load profiles
    profiles = np.load(args.input)

    # Convert to features
    features = profiles_to_features(profiles)

    # Cluster
    kmeans = KMeans(n_clusters=args.n, random_state=0, n_init=10).fit(features)
    labels = kmeans.labels_

    x_tsne = TSNE(n_components=2, n_jobs=-1, perplexity=500).fit_transform(features)

    for i in range(args.n):
        idx = np.where(labels == i)[0]
        plt.plot(x_tsne[idx, 0], x_tsne[idx, 1], '.')
    plt.show()

    # Save the labels
    np.save(args.labels, labels)
    np.save(args.tsne, x_tsne)
    verbose_print(args, f'Labels saved to {args.labels}')
    verbose_print(args, f't-SNE coordinates saved to {args.tsne}')

    # TODO: Save trained clustering model for classifying new samples (either KMeans or GaussianMixture)

    verbose_print(args, 'Calculating profiles done!')


def cluster_cli(subparsers):
    cluster_parser = subparsers.add_parser('cluster', help="Cluster cytoarchitectures",
                                           description='Clusters cytoarchtectures using ventricle profiles')
    cluster_parser.add_argument('input', help="Path to input profiles")
    cluster_parser.add_argument('labels', help="Path to output labels")
    cluster_parser.add_argument('tsne', help="Path to output t-SNE coordinates")
    cluster_parser.add_argument('-n', help="Number of clusters", type=int, default=5)
    cluster_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def classify_main(args):
    verbose_print(args, f'Training logistic model based on {args.profiles_train} and {args.labels_train}')

    # Load training data
    profiles_train = np.load(args.profiles_train)
    x_train = profiles_to_features(profiles_train)  # Normalizes the data (should we do this?)
    y_train = np.load(args.labels_train)
    classes = np.unique(y_train)

    # Train model
    clf = LogisticRegression(random_state=0,
                             solver='lbfgs',
                             multi_class='multinomial',
                             max_iter=200,
                             n_jobs=-1).fit(x_train, y_train)
    verbose_print(args, f'Training accuracy: {clf.score(x_train, y_train):.4f}')
    # verbose_print(args, f'Model coefficients:\n{clf.coef_}')
    # verbose_print(args, f'Model intercepts:\n{clf.intercept_}')

    # Apply classifier
    profiles = np.load(args.profiles)
    x = profiles_to_features(profiles)  # Scale the data
    labels = clf.predict(x)

    nb_cells = len(profiles)
    verbose_print(args, f'Classified {nb_cells} profiles into {len(classes)} cytoarchitecture classes')
    for c in classes:
        count = len(np.where(labels == c)[0])
        verbose_print(args, f'Class {c}: {count:10d} profiles {100 * count / nb_cells:10.3f}%')

    # Save the niche labels
    np.save(args.labels, labels)
    verbose_print(args, f'Labels saved to {args.labels}')

    verbose_print(args, f'Classifying done!')


def classify_cli(subparsers):
    classify_parser = subparsers.add_parser('classify', help="Cluster profiles into cyto labels",
                                            description='Clusters profiles into cytoarchitectures based on profiles')
    classify_parser.add_argument('profiles_train', help="Path to input profiles numpy array for training")
    classify_parser.add_argument('labels_train', help="Path to output cyto labels numpy array for training")
    classify_parser.add_argument('profiles', help="Path to input profiles numpy array to classify")
    classify_parser.add_argument('labels', help="Path to output cyto labels numpy array")
    classify_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cyto_cli(subparsers):
    cyto_parser = subparsers.add_parser('cyto', help="cytoarchitecture analysis",
                                        description="Organoid cytoarchitecture analysis")
    cyto_subparsers = cyto_parser.add_subparsers(dest='cyto_command', title='cyto subcommands')
    mesh_cli(cyto_subparsers)
    profiles_cli(cyto_subparsers)
    sample_cli(cyto_subparsers)
    cluster_cli(cyto_subparsers)
    classify_cli(cyto_subparsers)
    return cyto_parser


def cyto_main(args):
    commands_dict = {
        'mesh': mesh_main,
        'profiles': profiles_main,
        'sample': sample_main,
        'cluster': cluster_main,
        'classify': classify_main,
    }
    func = commands_dict.get(args.cyto_command, None)
    if func is None:
        print("Pickle Rick uses cyto subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'cyto', '-h'])
    else:
        func(args)
