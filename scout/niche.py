"""
Niche Module
=============

This module performs cytometry and neighborhood analysis.

"""

# Perform smoothing before sampling


# Sample using celltype.nuclei_centered_intensities and classify cell types


# Query neighbors


# Calculate proximities


# Define command-line functionality


def niche_cli(subparsers):
    niche_parser = subparsers.add_parser('niche', help="niche analysis",
                                         description='Organoid niche analysis tool')
    niche_parser.add_argument('centroids', help="Path to nuclei centroids")
    niche_parser.add_argument('images', help="Path to images to sample", nargs='+')
    niche_parser.add_argument('-g', help="Amount of gaussian smoothing", type=float, default=None)
    niche_parser.add_argument('-c', help="Chunk shape to use", type=int, nargs='+', default=None)
    niche_parser.add_argument('-w', help="Number of workers for use", type=int, default=None)
    return niche_parser


def niche_main(args):
    print('Running niche main')
    print(args)
