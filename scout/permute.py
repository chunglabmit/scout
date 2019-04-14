"""
Permute Module
===============

This module performs perumtation testing on multiscale organoid features

"""


# Define command-line functionality


def permute_cli(subparsers):
    permute_parser = subparsers.add_parser('permute', help="permutation testing",
                                           description="Permutation testing for multiscale organoid features")
    return permute_parser


def permute_main(args):
    print('Running permute main')
    print(args)
