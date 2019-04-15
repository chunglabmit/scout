"""
Cyto Module
============

This module performs organoid cytoarchitecture analysis

"""


# Define command-line functionality


def cyto_cli(subparsers):
    cyto_parser = subparsers.add_parser('cyto', help="cytoarchitecture analysis",
                                        description="Organoid cytoarchitecture analysis")
    cyto_parser.add_argument('segmentation', help="Path to ventricle segmentation image")
    cyto_parser.add_argument('labels', help="Path to cell type labels")

    return cyto_parser


def cyto_main(args):
    print('Running cyto main')
    print(args)


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
