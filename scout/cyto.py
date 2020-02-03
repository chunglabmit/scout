"""
Cyto Module
============

This module performs organoid cytoarchitecture analysis

These include the following subcommands:

    - mesh : compute surface mesh from segmentation
    - profiles : compute profiles along surface normals
    - sample : randomly sample profiles
    - combine : combine cytoarchitecture features from multiple organoids
    - cluster : cluster profiles into cytoarchitecture classes
    - classify : classify profiles into distinct cytoarchitectures
    - name : assign names to cytoarchitectures

"""
import os
import multiprocessing
import pickle
import subprocess
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.measure import marching_cubes_lewiner
from sklearn.preprocessing import scale
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import joblib
from umap import UMAP
import matplotlib.pyplot as plt
from scout.preprocess import gaussian_blur
from scout.niche import sample_main
from scout.niche import name_cli, name_main
from scout import io
from scout.utils import verbose_print, read_voxel_size, filter_points_in_box
try:
    if os.environ['DISPLAY'] == 'localhost:10.0':
        mlab = None
    else:
        from mayavi import mlab
except:
    pass

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


# Writes an .obj file for the output of marching cube algorithm.
# One = True for faces indexing starting at 1 as opposed to 0. Necessary for Blender/SurfIce

def write_obj(name, verts, faces, normals, values, one=False):
    """Write a .obj file for the output of marching cube algorithm.

    Parameters
    ----------
    name : str
        Ouput file name.
    verts : array
        Spatial coordinates for vertices as returned by skimage.measure.marching_cubes_lewiner().
    faces : array
        List of faces, referencing indices of verts as returned by skimage.measure.marching_cubes_lewiner().
    normals : array
        Normal direction of each vertex as returned by skimage.measure.marching_cubes_lewiner().
    one : bool
        Specify if faces values should start at 1 or at 0. Different visualization programs use different conventions.

    """
    if one:
        faces = faces + 1
    with open(name, 'w') as thefile:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))
        print("File written 30%")
        for item in normals:
            thefile.write("vn {0} {1} {2}\n".format(item[0], item[1], item[2]))
        print("File written 60%")
        for item in faces:
            thefile.write("f {0}//{0} {1}//{1} {2}//{2}\n".format(item[0], item[1], item[2]))


def write_point_cloud(name, verts):
    """Write a .obj file for a point cloud.

    Parameters
    ----------
    name : str
        Ouput file name.
    verts : array
        Spatial coordinates for vertices as returned by skimage.measure.marching_cubes_lewiner().

    """
    with open(name, 'w') as thefile:
        for item in verts:
            thefile.write("v {0} {1} {2}\n".format(item[0], item[1], item[2]))


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

    # Filter points within a box centered at the normal start
    start = vert - length - radius
    stop = vert + length + radius
    pts, idx = filter_points_in_box(pts, start, stop, return_idx=True)
    sox2_labels = sox2_labels[idx]
    tbr1_labels = tbr1_labels[idx]

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

def profiles_to_features(profiles, normalize=True):
    features = profiles.reshape((len(profiles), -1)).astype(np.float)  # Flattened profiles
    if normalize:
        features = scale(features)  # Normalize each feature (cell bin) to unit mean, zero variance
    return features


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
    verbose_print(args, f'Computed mesh with {len(normals)} normals')

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

    sox2_labels = labels[:, 0]
    tbr1_labels = labels[:, 1]

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
    profiles_parser.add_argument('-l', help='Length in micron of the profiles', type=float, default=300)
    profiles_parser.add_argument('-r', help='Radius of profile bins', type=int, default=25)
    profiles_parser.add_argument('-b', help='Number of bins in each profile', type=int, default=6)
    profiles_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    profiles_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def sample_cli(subparsers):
    sample_parser = subparsers.add_parser('sample', help="Randomly sample profiles",
                                          description='Randomly sample profiles before clustering')
    sample_parser.add_argument('samples', help="Number of samples to take", type=int)
    sample_parser.add_argument('index', help="Path to save sample index numpy array")
    sample_parser.add_argument('-i', '--inputs', help="Path to input numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-o', '--outputs', help="Path to sampled output numpy arrays", nargs='+', required=True)
    sample_parser.add_argument('-s', '--seed', help="Random seed", type=int, default=1)
    sample_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def combine_main(args):
    verbose_print(args, f'Combining profiles based on {args.input}')

    # Get full paths for sampled profiles from analysis CSV
    #parent_dir = os.path.abspath(os.path.join(args.input, os.pardir))
    #df = pd.read_csv(args.input, index_col=0)
    #paths = [os.path.join(parent_dir, df.loc[folder]['type'], folder, 'dataset', args.name) for folder in df.index]
    parent_dir = os.path.abspath(os.path.join(args.input, os.pardir, os.pardir))
    df = pd.read_csv(args.input)
    paths = [os.path.join(parent_dir, "datasets", smp, "cytoarchitecture", args.name) for smp in df["sample"]]

    # Adapted from niche.combine_main
    input_arrays = [np.load(path) for path in paths]
    combined = np.concatenate(input_arrays, axis=args.a)

    verbose_print(args, f'Saving combined features to {args.output} with shape {combined.shape}')
    np.save(args.output, combined)

    verbose_print(args, f'Saving organoid labels to {args.sample}')
    names = np.concatenate([i*np.ones(len(arr)) for i, arr in enumerate(input_arrays)])
    np.save(args.sample, names)

    verbose_print(args, f'Combining profiles done!')


def combine_cli(subparsers):
    combine_parser = subparsers.add_parser('combine', help="Combine sampled cytoarchitectures",
                                           description='Combine profiles from multiple organoids by concatenation')
    combine_parser.add_argument('input', help="Path to input analysis CSV")
    combine_parser.add_argument('-o', '--output', help="Path to output combined numpy array", required=True)
    combine_parser.add_argument('-s', '--sample', help="Path to output with sample name", required=True)
    combine_parser.add_argument('-n', '--name', help="Filename of sampled profiles", default='cyto_profiles_sample.npy')
    combine_parser.add_argument('-a', help="Axis to concatenate", type=int, default=0)
    combine_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cluster_main(args):
    # This is OLD... See the "determine cyto clusters" notebook

    verbose_print(args, f'Clustering profiles from {args.input}')

    # Load profiles
    profiles = np.load(args.input)

    # Convert to features
    features = profiles_to_features(profiles)

    # Cluster
    kmeans = KMeans(n_clusters=args.n, random_state=0, n_init=10).fit(features)
    labels = kmeans.labels_

    # x_tsne = TSNE(n_components=2, n_jobs=-1, perplexity=500).fit_transform(features)
    x_tsne = UMAP().fit_transform(features)

    if args.plot:
        for i in range(args.n):
            idx = np.where(labels == i)[0]
            plt.plot(x_tsne[idx, 0], x_tsne[idx, 1], '.', alpha=1.0, markersize=3)
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
    cluster_parser.add_argument('-p', '--plot', help="Plotting flag", action='store_true')
    cluster_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def classify_main(args):
    verbose_print(args, f'Training KNN model based on {args.profiles_train} and {args.labels_train}')

    # Load training data
    profiles_train = np.load(args.profiles_train)
    x_train = profiles_to_features(profiles_train, normalize=False)  # Normalizes the data (should we do this?)
    if args.umap is not None:
        model = joblib.load(args.umap)
        x_train = model.transform(x_train)
    y_train = np.load(args.labels_train)
    classes = np.unique(y_train)

    if args.load is None:
        verbose_print(args, f'Training new model')
        # Train model
        # Logistic regression model
        # clf = LogisticRegression(random_state=0,
        #                          solver='lbfgs',
        #                          multi_class='multinomial',
        #                          max_iter=200,
        #                          n_jobs=-1).fit(x_train, y_train)
        # KNN classifier
        clf = KNeighborsClassifier(n_neighbors=1)
        clf.fit(x_train, y_train)
        verbose_print(args, f'Training accuracy: {clf.score(x_train, y_train):.4f}')
    else:
        verbose_print(args, f'Loading model from {args.load}')
        clf = joblib.load(args.load)

    if args.save is not None:
        verbose_print(args, f'Saving model to {args.save}')
        joblib.dump(clf, args.save)

    # Apply classifier
    profiles = np.load(args.profiles)
    x = profiles_to_features(profiles, normalize=False)
    if args.umap is not None:
        x = model.transform(x)
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
    classify_parser.add_argument('--umap', help="Path to fit UMAP embedding", default=None)
    classify_parser.add_argument('--save', help="Path to save trained model", default=None)
    classify_parser.add_argument('--load', help="Path to load a trained model", default=None)
    classify_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cyto_cli(subparsers):
    cyto_parser = subparsers.add_parser('cyto', help="cytoarchitecture analysis",
                                        description="Organoid cytoarchitecture analysis")
    cyto_subparsers = cyto_parser.add_subparsers(dest='cyto_command', title='cyto subcommands')
    mesh_cli(cyto_subparsers)
    profiles_cli(cyto_subparsers)
    sample_cli(cyto_subparsers)
    combine_cli(cyto_subparsers)
    cluster_cli(cyto_subparsers)
    classify_cli(cyto_subparsers)
    name_cli(cyto_subparsers)
    return cyto_parser


def cyto_main(args):
    commands_dict = {
        'mesh': mesh_main,
        'profiles': profiles_main,
        'sample': sample_main,
        'combine': combine_main,
        'cluster': cluster_main,
        'classify': classify_main,
        'name': name_main,
    }
    func = commands_dict.get(args.cyto_command, None)
    if func is None:
        print("Pickle Rick uses cyto subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'cyto', '-h'])
    else:
        func(args)
