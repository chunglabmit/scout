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
