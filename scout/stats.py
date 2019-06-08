"""
Stats Module
=============

This module performs statistical analyses on sets of organoid features.

These include the following subcommands:

    - features : consolidate multiscale features with optional metadata
    - combine : combine features from multiple organoids
    - summarize : display an organoid summary

"""
import subprocess
import numpy as np
import pandas as pd
from scipy import stats
from scout.utils import verbose_print
import matplotlib.pyplot as plt


def ttest_main(args):
    verbose_print(args, 'Performing independent t-tests')

    # Read the combined features
    verbose_print(args, f'Loading group A from {args.a}')
    df_a = pd.read_excel(args.a)
    verbose_print(args, 'Group A table preview:')
    verbose_print(args, df_a.head())

    verbose_print(args, f'Loading group B from {args.b}')
    df_b = pd.read_excel(args.b)
    verbose_print(args, 'Group B table preview:')
    verbose_print(args, df_b.head())

    # Check that the features match between groups
    features_a = df_a['feature']
    features_b = df_b['feature']
    assert np.all(features_a == features_b)

    # Get values for each statistic from feature tables
    a = df_a.values[:, 1:].astype(np.float)
    b = df_b.values[:, 1:].astype(np.float)
    b = b * 2 * np.random.random(b.shape)  # Fake-data

    # Compute descriptive stats, including fold-changes
    a_means = a.mean(axis=1)
    a_stdev = a.std(axis=1)
    b_means = b.mean(axis=1)
    b_stdev = b.std(axis=1)
    fc = (b_means - a_means) / a_means

    # Compute p-values
    t, p = stats.ttest_ind(a, b, axis=1, equal_var=True, nan_policy='omit')  # Can 'propagate' nans too
    neg_logp = -np.log10(p)

    if args.plot:
        # Make volcano plot
        plt.plot(fc, neg_logp, 'k.', markersize=3)
        for i, feat in enumerate(features_a):
            if neg_logp[i] > 2:
                plt.annotate(feat, (fc[i], neg_logp[i]), fontsize=4)
        plt.show()

    # Save the fold-change, t-statistic, and p-values
    results = {
        'a_mean': a_means,
        'a_stdev': a_stdev,
        'b_means': b_means,
        'b_stdev': b_stdev,
        'fold-change': fc,
        't': t,
        'pvalue': p
    }
    df = pd.DataFrame(results, index=features_a)
    verbose_print(args, df.head())

    df.to_excel(args.output)

    verbose_print(args, f't-test done!')


def ttest_cli(subparsers):
    ttest_parser = subparsers.add_parser('t-test', help="Perform independent t-tests",
                                         description='Performs independent t-tests on two groups of organoid features')
    ttest_parser.add_argument('-a', help='Path to combined feature table for group A', required=True)
    ttest_parser.add_argument('-b', help='Path to combined feature table for group B', required=True)
    ttest_parser.add_argument('-o', '--output', help='Path to excel spreadsheet with results', required=True)
    ttest_parser.add_argument('-p', '--plot', help="Plotting flag", action='store_true')
    ttest_parser.add_argument('-v', '--verbose', help="Verbose flag", action='store_true')

# ========================


def stats_cli(subparsers):
    stats_parser = subparsers.add_parser('stats', help="statistical analyses",
                                         description="Perform statstical analyses on organoid features")
    stats_parser = stats_parser.add_subparsers(dest='stats_command', title='stats subcommands')
    ttest_cli(stats_parser)
    return stats_parser


def stats_main(args):
    commands_dict = {
        't-test': ttest_main,
    }
    func = commands_dict.get(args.stats_command, None)
    if func is None:
        print("Pickle Rick uses stats subcommands... be like Pickle Rick\n")
        subprocess.call(['scout', 'stats', '-h'])
    else:
        func(args)
