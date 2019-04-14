"""
Segment Module
===============

This module performs organoid regions segmentation.

"""


# Segment ventricles with U-Net


# Calculate local densities and threshold


# rasterize region labels


# Smooth with graph-cuts


# Define command-line functionality


def segment_cli(subparsers):
    segment_parser = subparsers.add_parser('segment', help="organoid region segmentation",
                                           description="Organoid region segmentation tool")
    segment_parser.add_argument('image', help="Path to nuclei image")
    segment_parser.add_argument('centroids', help="Path to nuclei centroids")
    segment_parser.add_argument('labels', help="Path to cell type labels")
    return segment_parser


def segment_main(args):
    print("Running segment main")
    print(args)
