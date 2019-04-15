"""
Multiscale Module
==================

This module builds the multiscale organoid feature vectors.

"""


# Define command-line functionality


def multiscale_cli(subparsers):
    multiscale_parser = subparsers.add_parser('multiscale', help="multiscale features",
                                              description="Build features for multiscale organoid analysis")
    return multiscale_parser


def multiscale_main(args):
    print('Running multiscale main')
    print(args)


"""

MULTISCALE
-----------
single-cell
cytoarchitecture
whole-organoid

"""

