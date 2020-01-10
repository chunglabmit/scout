Welcome to SCOUT's documentation!
=================================

.. image:: combined_render.jpg
    :width: 400px
    :align: center
    :alt: organoid_render

SCOUT is an open-source Python package for performing volumetric image analysis of intact cerebral organoids. [1]_
The details behind the development of this package can be found in the following publication:

.. [1]  Alex Albanese*, Justin Swaney*, Dae Hee Yun, Nicholas Evans, Jenna Antonucci-Johnson, Vincent Pham,
        Chloe Delepine, Mriganka Sur, Lee Gehrke, Kwanghun Chung. 3D Imaging and High Content Morphological
        Analysis of Intact Human Cerebral Organoids. **bioArxiv**, 2019.

If you use SCOUT, please be sure to cite this publication in your work.

The SCOUT package provides a command-line interface (CLI) for extracting multiscale features as well as a library of
tools that can be mixed-and-matched to build custom single-cell organoid analyses pipelines. SCOUT also includes
example Jupyter notebooks for users that prefer the more interactive, web-based development environment.

.. toctree::
   :maxdepth: 1
   :caption: Tutorial

   installation
   preprocessing
   single_cell
   proximity
   segmentation
   cytoarchitecture
   statistics

.. toctree::
   :maxdepth: 1
   :caption: Main Modules

   preprocess
   nuclei
   niche
   segment
   cyto
   multiscale

.. toctree::
   :maxdepth: 1
   :caption: Additional Modules

   io
   utils
   curvature
   detection
   score

.. toctree::
   :maxdepth: 1
   :caption: Cheatsheet Modules
  
   analysis_cheatsheet
   

Contributing
=============

For those who want to add additional functionality to the existing SCOUT pipeline, a pull request can be submitted
to the SCOUT repo. Feature requests can be submitted as issues on the SCOUT repo as well.

Contact
========

If you have questions about SCOUT or how to use it, please submit an issue to the SCOUT Github repo. We welcome
feedback from the organoid scientific community.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

This site is maintained by members of the `Chung Lab`_ at MIT.

.. _Chung Lab: http://www.chunglab.org/
