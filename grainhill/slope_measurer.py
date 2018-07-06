#!/usr/bin/env python2
# -*- cpding: utf-8 -*-
"""
slope_measurer.py: GrainHill module that retrieves the (x, z) coordinates of
surface bedrock cells in a cellular hillslope cross-section.

Created Wed Apr 18 14:48 2018

@author: SiccarPoint
"""

import numpy as np


class SlopeMeasurer(object):
    """SlopeMeasurer: extracts nodes at the bedrock-air or regolith-air
    surface, and fits to this data according to user choices."""

    def __init__(self, grain_hill_model, pick_only_rock=True, rock_id=8,
                 static_regolith_id=7, air_id=0):
        """Initialise a SlopeMeasurer.

        If pick_only_rock, SlopeMeasurer will track the bedrock-air interface.
        If False, it will track the regolith-air interface.
        """
        self.model = grain_hill_model
        self.grid = grain_hill_model.ca.grid
        self.rock_id = rock_id
        self.air_id = air_id
        self.reg_id = static_regolith_id
        self.bedrock_surface = []

        lsd = self.model.ca.link_state_dict
        if pick_only_rock:
            rock_options = (rock_id, )
        else:
            rock_options = (rock_id, static_regolith_id)
        state_options = lsd.keys()
        airrock_states = []
        self.rockend = []
        for opt in state_options:
            if opt[0] in rock_options:
                if opt[1] is air_id:
                    airrock_states.append(opt)
                    self.rockend.append(0)
            elif opt[0] is air_id:
                if opt[1] in rock_options:
                    airrock_states.append(opt)
                    self.rockend.append(1)
        self.airrock_link_state_codes = [
            lsd[state] for state in airrock_states]

    def pick_rock_surface(self):
        """Identify nodes at the rock-air (or regolith-air, as appropriate)
        interface, and return the IDs of the rock/regolith nodes as an array.

        Examples
        --------
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((5, 7), show_plots=False)
        >>> gh.ca.node_state[:18] = 8
        >>> gh.ca.node_state[19:21] = 7  # a partial soil mantle
        >>> gh.ca.node_state[22] = 8  # a rock nubbin
        >>> gh.ca.assign_link_states_from_node_types()
        >>> sm = SlopeMeasurer(gh)
        >>> sm.pick_rock_surface()
        array([11, 14, 15, 16, 22])
        >>> sm = SlopeMeasurer(gh, False)
        >>> sm.pick_rock_surface()
        array([11, 14, 15, 16, 19, 20, 22])
        """
        # find the links that are air-rock (or air-reg)
        rock_nodes = []
        for statecode, end in zip(self.airrock_link_state_codes, self.rockend):
            is_airrock = self.model.ca.link_state == statecode
            if end is 0:
                rock_nodes.extend(self.grid.node_at_link_tail[is_airrock])
            else:
                rock_nodes.extend(self.grid.node_at_link_head[is_airrock])
        self.exposed_surface = np.array(list(set(rock_nodes)))
        self.exposed_surface.sort()

        return self.exposed_surface

    def calc_coords_of_surface_points(self, min_x=None, max_x=None,
                                      first_nodes=None):
        """
        Calculate the x and z coordinates of a surface which has already been
        found with pick_rock_surface(). Saves the surface nodes extracted as
        the property `nodes_fitted`. Assumes `pick_rock_surface` has already
        been run.

        Parameters
        ----------
        min_x : float or None
            If float, only take nodes in positive x-direction from that point
        max_x : float or None
            If float, only take nodes in negative x-direction from that point
        first_nodes : int or None
            If int, take only the first int nodes (by node ID). Can be combined
            with other params, giving the minimum possible number of nodes.

        Returns
        -------
        surface_x, surface_z : arrays of floats
            The coordinates of the selected surface points

        Examples
        --------
        >>> import numpy as np
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((4, 8), show_plots=False)
        >>> gh.ca.node_state[:16] = 8
        >>> gh.ca.node_state[17] = 8  # make the surface imperfect
        >>> gh.ca.assign_link_states_from_node_types()
        >>> sm = SlopeMeasurer(gh)
        >>> sm.pick_rock_surface()
        array([10, 11, 12, 13, 14, 15, 17])
        >>> x, z = np.round(sm.calc_coords_of_surface_points(first_nodes=6), 2)
        >>> x
        array([ 3.46,  5.2 ,  0.87,  2.6 ,  4.33,  6.06])
        >>> z
        array([ 1. ,  1. ,  1.5,  1.5,  1.5,  1.5])
        """
        surface_x = self.grid.node_x[self.exposed_surface]
        surface_z = self.grid.node_y[self.exposed_surface]
        cond = np.ones(len(surface_x), dtype=bool)
        if min_x is not None:
            if max_x is not None:
                cond = np.logical_and(surface_x > min_x, surface_x < max_x)
            else:
                cond = surface_x > min_x
            surface_x = surface_x[cond]
            surface_z = surface_z[cond]
        elif max_x is not None:
            cond = surface_x < max_x
            surface_x = surface_x[cond]
            surface_z = surface_z[cond]

        # now see if we need to truncate:
        if first_nodes is not None:
            surface_x = surface_x[:first_nodes]
            surface_z = surface_z[:first_nodes]
            cond = np.where(cond)[0]
            cond = cond[:first_nodes]

        # save the nodes we used
        self.nodes_fitted = self.exposed_surface[cond]

        return surface_x, surface_z

    def fit_straight_line_to_coords(self, surface_x, surface_z):
        """Fits a planar surface to a set of provided (surface) x, z points.

        Parameters
        ----------
        surface_x : array of float
            The x coordinates of the points to fit
        surface_z : array of float
            The z coordinates of the points to fit

        Returns
        -------
        array([m, c]) : floats
            The surface gradient (m) and intersect at x=0 (c).

        Examples
        --------
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((4, 8), show_plots=False)
        >>> gh.ca.node_state[:16] = 8
        >>> gh.ca.node_state[17] = 8  # make the surface imperfect
        >>> gh.ca.assign_link_states_from_node_types()
        >>> sm = SlopeMeasurer(gh)
        >>> sm.pick_rock_surface()
        array([10, 11, 12, 13, 14, 15, 17])
        >>> x, z = sm.calc_coords_of_surface_points()
        >>> m_and_c = sm.fit_straight_line_to_coords(x, z)
        >>> round(m_and_c[0], 3)  # the gradient
        -0.082
        >>> round(m_and_c[1], 3)  # the intersect
        1.714

        """
        # fit the line
        polyparams = np.polyfit(surface_x, surface_z, 1)

        return polyparams

    def fit_straight_line_to_surface(self, min_x=None, max_x=None,
                                     first_nodes=None):
        """
        Fit a straight line to a surface which has already been found with
        pick_rock_surface().

        Saves the internal params m, c, S ("slope", +ve m), and dip_angle
        (from horizontal, in **positive degrees**).

        Parameters
        ----------
        min_x : float or None
            If float, only take nodes in positive x-direction from that point
        max_x : float or None
            If float, only take nodes in negative x-direction from that point
        first_nodes : int or None
            If int, take only the first int nodes (by node ID). Can be combined
            with other params, giving the minimum possible number of nodes.

        Returns
        -------
        array([m, c]) : floats
            The surface gradient (m) and intersect at x=0 (c).

        Examples
        --------
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((4, 8), show_plots=False)
        >>> gh.ca.node_state[:16] = 8
        >>> gh.ca.node_state[17] = 8  # make the surface imperfect
        >>> gh.ca.assign_link_states_from_node_types()
        >>> sm = SlopeMeasurer(gh)
        >>> sm.pick_rock_surface()
        array([10, 11, 12, 13, 14, 15, 17])
        >>> m_and_c = sm.fit_straight_line_to_surface()
        >>> round(m_and_c[0], 3)  # the gradient
        -0.082
        >>> round(sm.S, 3)
        0.082
        >>> round(sm.dip_angle, 2)
        4.72

        ...and without the bump...

        >>> m_and_c = sm.fit_straight_line_to_surface(min_x=1.8)
        >>> sm.nodes_fitted
        array([10, 11, 13, 14, 15])
        >>> np.isclose(m_and_c[0], 0.)
        True

        Then an alternative way:

        >>> m_and_c = sm.fit_straight_line_to_surface(min_x=1.7,
        ...                                           first_nodes=5)
        >>> sm.nodes_fitted  # 17 gets chopped by first_nodes==5
        array([10, 11, 13, 14, 15])
        >>> np.isclose(sm.dip_angle, 0.)
        True
        """
        surface_x, surface_z = self.calc_coords_of_surface_points(
            min_x, max_x, first_nodes)
        polyparams = self.fit_straight_line_to_coords(surface_x, surface_z)

        self.m = polyparams[0]
        self.c = polyparams[1]
        self.S = np.abs(self.m)
        self.dip_angle = np.arctan(self.S)*180./np.pi

        return polyparams


if __name__ == "__main__":
    import doctest
    doctest.testmod()
