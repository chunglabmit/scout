"""
Multiscale Module
==================

This module builds the multiscale organoid feature vectors

These include the following subcommands:
    - metadata : attach metadata to multiscale features
    - features : consolidate multiscale features with optional metadata
    - summarize : display an organoid summary

"""

import subprocess
from scout.utils import verbose_print


# TODO: Allow clustering facilities to sample from multiple organoids

# Define command-line functionality

def features_main(args):
    verbose_print(args, f'Consolidating multiscale features')

    verbose_print(args, f'Multiscale features done!')


def features_cli(subparsers):
    cluster_parser = subparsers.add_parser('features', help="Consolidate multiscale features",
                                           description='Consolidate multiscale features for an organoid')
    cluster_parser.add_argument('input', help="Path to input profiles")
    cluster_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')


def multiscale_cli(subparsers):
    multiscale_parser = subparsers.add_parser('multiscale', help="multiscale features",
                                              description="Build features for multiscale organoid analysis")
    multiscale_parser = multiscale_parser.add_subparsers(dest='multiscale_command', title='multiscale subcommands')
    features_cli(multiscale_parser)
    return multiscale_parser


def multiscale_main(args):
    commands_dict = {
        'features': features_main,
    }
    func = commands_dict.get(args.multiscale_command, None)
    if func is None:
        print("Pickle Rick uses multiscale subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'multiscale', '-h'])
    else:
        func(args)
