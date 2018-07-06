#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
cosmogenic_irradiator.py: GrainHill module that calculates input of cosmogenic
nuclide concentration to regolith and rock particles.

Created on Sat Apr 14 09:14:15 2018

@author: gtucker
"""

import numpy as np

def row_col_to_id(row, col, num_cols):
    """Return ID for node at given row and column.

    Examples
    --------
    >>> row_col_to_id(0, 1, 8)
    4
    >>> row_col_to_id(0, 6, 8)
    3
    >>> row_col_to_id(0, 1, 5)
    3
    >>> row_col_to_id(2, 3, 5)
    14
    """
    return row * num_cols + col // 2 + (col % 2) * ((num_cols + 1) // 2)

class CosmogenicIrradiator(object):
    """CosmogenicIrradiator: handles addition of cosmogenic nuclide content
    to non-fluid cells in GrainHill model."""

    def __init__(self, grain_hill_model, prod_rate, decay_depth):
        """Initialize a CosmogenicIrradiator with production rate and decay
        depth."""
        self.model = grain_hill_model
        self.grid = grain_hill_model.ca.grid
        self.prod_rate = prod_rate
        self.decay_depth = decay_depth

        # Cosmo field: create it, or get ref to it if it already exists
        name = 'cosmogenic_nuclide__concentration'
        if name in grain_hill_model.ca.grid.at_node:
            self.cosmo = grain_hill_model.ca.grid.at_node[name]
        else:
            self.cosmo = grain_hill_model.ca.grid.add_zeros('node', name)

    def add_cosmos(self, duration, cell_width=1.0):
        """Add cosmogenic nuclide content to grains.

        Notes
        -----
        For each inner column
            cumulative_depth = half of one cell
            for each row from top to bottom
                if cell is not fluid
                    add cosmo based on cumulative depth
                    increment cumulative depth by one cell height

        Examples
        --------
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((3, 5), show_plots=False)
        >>> gh.ca.node_state
        array([8, 7, 8, 7, 7, 0, 7, 0, 7, 7, 0, 0, 0, 0, 0])
        >>> ci = CosmogenicIrradiator(gh, 1.0, 2.0)
        >>> ci.cosmo
        array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
        >>> ci.add_cosmos(1.0)
        >>> np.round(ci.cosmo, 3)
        array([0.   , 0.472, 0.   , 0.472, 0.472, 0.   , 0.779, 0.   , 0.779,
               0.779, 0.   , 0.   , 0.   , 0.   , 0.   ])
        """
        for c in range(1, self.grid.number_of_node_columns - 1):
            cumulative_depth = cell_width / 2.0
            for r in range(self.grid.number_of_node_rows - 1, -1, -1):
                node_id = row_col_to_id(r, c, self.grid.number_of_node_columns)
                if self.model.ca.node_state[node_id] != 0:
                    dose_rate = self.prod_rate * np.exp(-cumulative_depth
                                                        / self.decay_depth)
                    self.cosmo[self.model.ca.propid[node_id]] += (dose_rate
                                                                  * duration)
                    cumulative_depth += cell_width
                else:
                    self.cosmo[self.model.ca.propid[node_id]] = 0.0
