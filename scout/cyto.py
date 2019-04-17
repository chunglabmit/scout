"""
Cyto Module
============

This module performs organoid cytoarchitecture analysis

These include the following subcommands:
    - smooth : compute radial profiles of each cell-type

"""

import pickle
import subprocess
import numpy as np
from skimage.measure import marching_cubes_lewiner
from mayavi import mlab
from scout.preprocess import gaussian_blur
from scout import io
from scout.utils import verbose_print


# Smoothing

def smooth_segmentation(seg, sigma=1, scale_factor=10):
    binary = (seg > 0)
    smooth = scale_factor * gaussian_blur(binary, sigma)
    return smooth.astype(np.float32)


# Meshing

def marching_cubes(seg, level, spacing, step_size):
    return marching_cubes_lewiner(seg, level=level, spacing=spacing, step_size=step_size, allow_degenerate=False)


def save_mesh(path, mesh):
    with open(path, 'wb') as f:
        pickle.dump(mesh, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_mesh(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_mesh(verts, faces, color=(1, 0, 0), figure=None):
    if figure is not None:
        mlab.figure(figure)
    return mlab.triangular_mesh([vert[0] for vert in verts],
                                [vert[1] for vert in verts],
                                [vert[2] for vert in verts],
                                faces,
                                color=color)


# Define command-line functionality

def smooth_main(args):
    verbose_print(args, f'Smoothing segmentation at {args.input}')

    # Load the segmentation
    seg = io.imread(args.input)

    # Smooth the segmentation to float
    smoothed = smooth_segmentation(seg, args.g, args.f)

    # Save the result
    io.imsave(args.output, smoothed, compress=3)
    verbose_print(args, f'Smoothed segmentation saved to {args.output}')

    verbose_print(args, 'Smoothing done!')


def smooth_cli(subparsers):
    smooth_parser = subparsers.add_parser('smooth', help="Smooth a segmentation",
                                          description='Smooth a binary segmentation to float')
    smooth_parser.add_argument('input', help="Path to input segmentation TIFF")
    smooth_parser.add_argument('output', help="Path to output smoothed segmentation TIFF")
    smooth_parser.add_argument('-g', help="Amount of gaussian smoothing", type=float, nargs='+', default=1.0)
    smooth_parser.add_argument('-f', help="Scale factor for smoothed segmentation", type=float, default=1.0)
    smooth_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def mesh_main(args):
    verbose_print(args, f'Meshing segmentation at {args.input} at spacing {args.spacing} and step size {args.step}')

    # Load segmentation
    seg = io.imread(args.input)

    # Calculate mesh surface
    verts, faces, normals, values = marching_cubes(seg, args.l, args.spacing, args.step)
    mesh = {
        'verts': verts,
        'faces': faces,
        'normals': normals,
        'values': values
    }

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
    mesh_parser.add_argument('output', help="Path to output mesh")
    mesh_parser.add_argument('-l', help='Isolevel for surface', type=float, default=0.2)
    # Spacing is really the voxel size accounting for downsampling too
    mesh_parser.add_argument('--spacing', help="Voxel spacing", type=float, nargs='+', default=(1, 1, 1))
    mesh_parser.add_argument('--step', help="Step size", type=int, default=1)
    mesh_parser.add_argument('-p', '--plot', help="Flag to show plot", action='store_true')
    mesh_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def cyto_cli(subparsers):
    cyto_parser = subparsers.add_parser('cyto', help="cytoarchitecture analysis",
                                        description="Organoid cytoarchitecture analysis")
    cyto_subparsers = cyto_parser.add_subparsers(dest='cyto_command', title='segment subcommands')
    smooth_cli(cyto_subparsers)
    mesh_cli(cyto_subparsers)
    return cyto_parser


def cyto_main(args):
    commands_dict = {
        'smooth': smooth_main,
        'mesh': mesh_main,
    }
    func = commands_dict.get(args.cyto_command, None)
    if func is None:
        print("Pickle Rick uses cyto subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'cyto', '-h'])
    else:
        func(args)


"""

CYTOARCHITECTURE
-----------------
compute-normals
    ventricle seg -> mesh
sample-normals
    mesh -> subset of normals
compute-profiles
    subset of normal + centroids + cell-type labels -> profiles
cluster-profiles
    profiles -> tSNE + cytoarchitecture labels
classify-niches
    profiles + subset of cytoarchitecture labels -> train logistic model -> model weights + all normal labels

"""
