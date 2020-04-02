# SCOUT (Single-cell and Cytoarchitecture Analysis of Organoids using Unbiased Techniques)
[![Build Status](https://travis-ci.org/chunglabmit/scout.svg?branch=master)](https://travis-ci.org/chunglabmit/scout)
[![Docs](https://img.shields.io/badge/docs-0.1.0a-brightgreen.svg)](https://chunglabmit.github.io/scout/)

**Multiscale hyperdimensional phenotypic analysis of organoids

Please check doc/index.html for more detailed documentation on how to get
started with SCOUT. Thank you.

## System requirements

SCOUT has been tested estensively on a variety of Linux workstations with the
following minimum specifications:

- 32 GB RAM
- 1 TB HDD
- 4-core i7 CPU
- Ubuntu 16.04 LTS or 18.04 LTS

Windows and Mac users may be able to install from source, but this has not been
tested. On these platforms, Docker may be an easier option for installing and
running SCOUT.

SCOUT was tested on Python 3.6 and 3.7. All of the required dependencies are
listed in the _requirements.txt_ file, and additional developement dependencies
can be found in the _environment.yml_ file for recreating a conda environment.
Specific versions that were used for running SCOUT are also described in the
_specific\_versions.txt_ file.

## Installation guide

A guide for installing SCOUT is available [here](https://chunglabmit.github.io/scout/installation.html). Typically, takes ~15
minutes to install SCOUT on a Linux desktop computer.

## Demo

The SCOUT [documentation](https://chunglabmit.github.io/scout/index.html) has a
"Tutorial" section to show users how to use SCOUT. After installation, the
tutorial is acts as a pipeline walkthrough for a typical two-group organoid
analysis (treatment vs control). The expected runtime for the SCOUT pipeline on
a single organoid is ~4 hours, and the overall runtime on a comparative study
will scale with the number of organoids.

## Reproduction

The `notebooks/` folder contains Jupyter notebooks with all of the custom code
used to generate visualzations for presenting the main findings in the SCOUT
paper. After installing SCOUT, running `jupyter notebook` within the
`notebooks/` folder will allow you to read and run these notebooks.


