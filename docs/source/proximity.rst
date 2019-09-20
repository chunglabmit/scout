Spatial sub-catergorization
=============================

Given the physical coordinates of all nuclei as well as individual cell-type labels, SCOUT aims to furher sub-categorize
cells based on their spatial context.

The intuition behind this type of analysis is that although cerebral organoids
do not have consistent a overall shape and region segmentation, they do self-assemble into regions that can be
identified by staining for characteristic proteins, such as SOX2 (neural progenitors) and TBR1 (post-mitotic neurons).
The position of a cell relative to these regional markers can provide useful information for all cells, even for those
that stained negatively for the regional markers.

This type of spatial proximity analysis would also be useful in
studies that look at how certain labeled cell populations tend to arrange relative to other cell types.

Calculate Proximities
----------------------

The nuclei physical coordinates (in micron) and the cell-type labels for each cell are used to compute the
"proximity" to each cell-type using the following command:

.. code-block:: bash

   scout niche proximity centroids_um.npy nuclei_gating.npy niche_proximities.npy -r 25 25 -v -p -k 2

This command creates a new numpy array *niche_proximities.npy*, which contains proximity values in the range [0, 1]
for all cells and each cell-type. The *-r 25 25* argument specifies 25 micron reference distance for what is considered
"close", and the *-k 2* argument specifies how many nearest neighbors to consider in the proximity to each cell-type.

If the *-p* flag is specified, then a scatter plot of proximities will be shown. Typically, cells with very high
proximity to any given cell-type are also positive for that cell-type marker. However, cells that stained negatively
may start to separate into four smaller quadrants (assuming there are two regional markers).

Gate Proximities
-----------------

Assuming that two regional markers are given, we can gate the proximity scatter plot into 6 main regions. In the case
of SOX2 and TBR1 staining, these regions would correspond to core double negative cells (DN), SOX2+ cells (SOX2),
TBR1 cells (TBR1), double positive cells (DP), TBR1-adjacent cells (TBR1Adj), SOX2-adjacent cells (SOX2Adj), and
Co-adjacent cells (CoAdj). We can name these groups using the following command:

.. code-block:: bash

   scout niche name DN SOX2 TBR1 DP TBR1Adj SOX2Adj CoAdj -o niche_names.csv -v

To set the gates and obtain labels for each cell, we then run the following command:

.. code-block:: bash

    scout niche gate niche_proximities.npy niche_labels.npy --low 0.35 0.30 --high 0.66 0.63 -p -v --alpha 0.01

where *niche_labels.npy* is a newly created numpy array containing a unique integer label for each cell corresponding
to the names in *niche_names.csv*. The arguments *--low 0.35 0.30* and *--high 0.66 0.63* specify the cutoffs for the
proximity gating. In this case, the SOX2 proximity gates would be set at 0.35 and 0.66, and the TBR1 proximity gates
would be set at 0.30 and 0.63.
