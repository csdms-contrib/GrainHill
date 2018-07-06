#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
block_hill.py: version of GrainHill that adds blocks.

Created on Sat Jun 24 10:54:53 2017

@author: gtucker
"""

from grainhill import GrainHill
from lattice_grain import (lattice_grain_node_states,
                           lattice_grain_transition_list)
from landlab.ca.boundaries.hex_lattice_tectonicizer import LatticeUplifter
from landlab.ca.celllab_cts import Transition

BLOCK_ID = 9


class BlockHill(GrainHill):
    """
    Model hillslope evolution with 'block' particles that can be undermined
    and weathered but not disturbed/activated.
    """
    def __init__(self, grid_size, report_interval=1.0e8, run_duration=1.0, 
                 output_interval=1.0e99, settling_rate=2.2e8,
                 disturbance_rate=1.0, weathering_rate=1.0, 
                 uplift_interval=1.0, plot_interval=1.0e99, friction_coef=0.3,
                 rock_state_for_uplift=7, opt_rock_collapse=False,
                 block_layer_dip_angle=0.0, block_layer_thickness=1.0,
                 layer_left_x=0.0, y0_top=0.0,
                 show_plots=True, **kwds):
        """Call the initialize() method."""
        self.initialize(grid_size, report_interval, run_duration,
                        output_interval, settling_rate, disturbance_rate,
                        weathering_rate, uplift_interval, plot_interval,
                        friction_coef, rock_state_for_uplift,
                        opt_rock_collapse, block_layer_dip_angle,
                        block_layer_thickness, layer_left_x, y0_top,
                        show_plots, **kwds)

    def initialize(self, grid_size, report_interval, run_duration,
                   output_interval, settling_rate, disturbance_rate,
                   weathering_rate, uplift_interval, plot_interval,
                   friction_coef, rock_state_for_uplift, opt_rock_collapse,
                   block_layer_dip_angle, block_layer_thickness, layer_left_x,
                   y0_top, show_plots, **kwds):
        """Initialize the BlockHill model."""

        # Set block-related variables
        self.block_layer_dip_angle = block_layer_dip_angle
        self.block_layer_thickness = block_layer_thickness
        self.layer_left_x = layer_left_x
        self.y0_top = y0_top

        # Call parent class init
        super(BlockHill, self).__init__(grid_size=grid_size, 
                                          report_interval=report_interval, 
                                          run_duration=run_duration,
                                          output_interval=output_interval,
                                          settling_rate=settling_rate,
                                          disturbance_rate=disturbance_rate,
                                          weathering_rate=weathering_rate,
                                          uplift_interval=uplift_interval,
                                          plot_interval=plot_interval,
                                          friction_coef=friction_coef,
                                          rock_state_for_uplift=rock_state_for_uplift,
                                          opt_rock_collapse=opt_rock_collapse,
                                          show_plots=show_plots, **kwds)

        self.uplifter = LatticeUplifter(self.grid, 
                                self.grid.at_node['node_state'],
                                opt_block_layer=True,
                                block_ID=8,
                                block_layer_dip_angle=block_layer_dip_angle,
                                block_layer_thickness=block_layer_thickness,
                                layer_left_x=layer_left_x, y0_top=y0_top)

    def node_state_dictionary(self):
        """
        Create and return dict of node states.
        
        Overrides base-class method. Here, we call on a function in
        the lattice_grain module, and then add an additional state for blocks.
        """
        nsd = lattice_grain_node_states()
        nsd[BLOCK_ID] = 'block'
        return nsd

    def add_weathering_and_disturbance_transitions(self, xn_list, d=0.0, w=0.0,
                                                   collapse_rate=0.0):
        """
        Add transition rules representing weathering and/or grain disturbance
        to the list, and return the list. Overrides method of same name in
        GrainHill.
        
        Parameters
        ----------
        xn_list : list of Transition objects
            List of objects that encode information about the link-state 
            transitions. Normally should first be initialized with lattice-grain
            transition rules, then passed to this function to add rules for
            weathering and disturbance.
        d : float (optional)
            Rate of transition (1/time) from fluid / resting grain pair to
            mobile-grain / fluid pair, representing grain disturbance.
        w : float (optional)
            Rate of transition (1/time) from fluid / rock pair to
            fluid / resting-grain pair, representing weathering.
        
        Returns
        -------
        xn_list : list of Transition objects
            Modified transition list.
        """

        # Disturbance rule
        if d > 0.0:
            xn_list.append( Transition((7,0,0), (0,1,0), d, 'disturbance') )
            xn_list.append( Transition((7,0,1), (0,2,1), d, 'disturbance') )
            xn_list.append( Transition((7,0,2), (0,3,2), d, 'disturbance') )
            xn_list.append( Transition((0,7,0), (4,0,0), d, 'disturbance') )
            xn_list.append( Transition((0,7,1), (5,0,1), d, 'disturbance') )
            xn_list.append( Transition((0,7,2), (6,0,2), d, 'disturbance') )

        # Weathering rule
        if w > 0.0:
            xn_list.append( Transition((8,0,0), (BLOCK_ID,0,0), w, 'weathering') )
            xn_list.append( Transition((8,0,1), (BLOCK_ID,0,1), w, 'weathering') )
            xn_list.append( Transition((8,0,2), (BLOCK_ID,0,2), w, 'weathering') )
            xn_list.append( Transition((0,8,0), (0,BLOCK_ID,0), w, 'weathering') )
            xn_list.append( Transition((0,8,1), (0,BLOCK_ID,1), w, 'weathering') )
            xn_list.append( Transition((0,8,2), (0,BLOCK_ID,2), w, 'weathering') )

            # "Vertical rock collapse" rule: a rock particle overlying air
            # will collapse, transitioning to a downward-moving grain
            if collapse_rate > 0.0:
                xn_list.append( Transition((0,8,0), (0,BLOCK_ID,0), collapse_rate,
                                           'rock collapse'))

#        if _DEBUG:
#            print
#            print 'setup_transition_list(): list has',len(xn_list),'transitions:'
#            for t in xn_list:
#                print '  From state',t.from_state,'to state',t.to_state,'at rate',t.rate,'called',t.name

        return xn_list

    def add_block_transitions(self, xn_list):
        """Adds transitions for block undermining and weathering."""

        # Undermining
        xn_list.append( Transition((0,BLOCK_ID,0), (BLOCK_ID,0,0),
                                   self.settling_rate, 'block settling') )
        xn_list.append( Transition((0,BLOCK_ID,1), (BLOCK_ID,0,1),
                                   self.disturbance_rate, 'block settling') )
        xn_list.append( Transition((BLOCK_ID,0,2), (0,BLOCK_ID,2),
                                   self.disturbance_rate, 'block settling') )

        # Weathering
        xn_list.append( Transition((0,BLOCK_ID,0), (0,7,0),
                                   self.weathering_rate, 'block weathering') )
        xn_list.append( Transition((BLOCK_ID,0,0), (7,0,0),
                                   self.weathering_rate, 'block weathering') )
        xn_list.append( Transition((0,BLOCK_ID,1), (0,7,1),
                                   self.weathering_rate, 'block weathering') )
        xn_list.append( Transition((BLOCK_ID,0,1), (7,0,1),
                                   self.weathering_rate, 'block weathering') )
        xn_list.append( Transition((0,BLOCK_ID,2), (0,7,2),
                                   self.weathering_rate, 'block weathering') )
        xn_list.append( Transition((BLOCK_ID,0,2), (7,0,2),
                                   self.weathering_rate, 'block weathering') )

        # Collision w block
        xn_list.append( Transition((1,BLOCK_ID,0), (7,BLOCK_ID,0),
                                   self.settling_rate, 'hit block') )
        xn_list.append( Transition((2,BLOCK_ID,1), (7,BLOCK_ID,1),
                                   self.settling_rate, 'hit block') )
        xn_list.append( Transition((3,BLOCK_ID,2), (7,BLOCK_ID,2),
                                   self.settling_rate, 'hit block') )
        xn_list.append( Transition((BLOCK_ID,4,0), (BLOCK_ID,7,0),
                                   self.settling_rate, 'hit block') )
        xn_list.append( Transition((BLOCK_ID,5,1), (BLOCK_ID,7,1),
                                   self.settling_rate, 'hit block') )
        xn_list.append( Transition((BLOCK_ID,6,2), (BLOCK_ID,7,2),
                                   self.settling_rate, 'hit block') )

        return xn_list

    def transition_list(self):
        """
        Make and return list of Transition object.
        """
        xn_list = lattice_grain_transition_list(g=self.settling_rate,
                                                f=self.friction_coef,
                                                motion=self.settling_rate)
#        xn_list = super(BlockHill, self).add_weathering_and_disturbance_transitions(xn_list,
#                    self.disturbance_rate, self.weathering_rate,
#                    collapse_rate=self.collapse_rate)
        xn_list = self.add_weathering_and_disturbance_transitions(xn_list,
                    self.disturbance_rate, self.weathering_rate,
                    collapse_rate=self.collapse_rate)
        
        xn_list = self.add_block_transitions(xn_list)
        return xn_list
        
    def initialize_node_state_grid(self):
        """Set up initial node states.

        Examples
        --------
        >>> bh = BlockHill((5, 7))
        >>> bh.grid.at_node['node_state']        
        array([9, 7, 7, 9, 7, 7, 7, 0, 7, 7, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        """

        # For shorthand, get a reference to the node-state grid
        nsg = self.grid.at_node['node_state']

        # Fill the bottom two rows with grains
        right_side_x = 0.866025403784 * (self.grid.number_of_node_columns - 1)
        for i in range(self.grid.number_of_nodes):
            if self.grid.node_y[i] < 2.0:
                if (self.grid.node_x[i] > 0.0 and
                    self.grid.node_x[i] < right_side_x):
                    nsg[i] = 7
        
        # Place "wall" particles in the lower-left and lower-right corners
        if self.grid.number_of_node_columns % 2 == 0:
            bottom_right = self.grid.number_of_node_columns - 1
        else:
            bottom_right = self.grid.number_of_node_columns // 2
        nsg[0] = BLOCK_ID  # bottom left
        nsg[bottom_right] = BLOCK_ID
        
        return nsg


if __name__ == '__main__':
    bh = BlockHill(grid_size=(3, 3))

        
