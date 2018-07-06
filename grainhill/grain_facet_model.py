#!/usr/env/python
"""
Model of normal-fault facet evolution using CTS lattice grain approach.
"""

import sys
from grainhill import CTSModel, plot_hill
from grainhill.lattice_grain import (lattice_grain_node_states,
                                     lattice_grain_transition_list)
import time
import numpy as np
from landlab import CLOSED_BOUNDARY
from landlab.ca.celllab_cts import Transition
from landlab.ca.boundaries.hex_lattice_tectonicizer import LatticeNormalFault


SECONDS_PER_YEAR = 365.25 * 24 * 3600
_DEBUG = False


class GrainFacetSimulator(CTSModel):
    """
    Model facet-slope evolution with 60-degree normal-fault slip.
    """
    def __init__(self, grid_size, report_interval=1.0e8, run_duration=1.0, 
                 output_interval=1.0e99, disturbance_rate=0.0,
                 weathering_rate=0.0, dissolution_rate=0.0,
                 uplift_interval=1.0, plot_interval=1.0e99, friction_coef=0.3,
                 fault_x=1.0, cell_width=1.0, grav_accel=9.8,
                 plot_file_name=None, **kwds):
        """Call the initialize() method."""
        self.initialize(grid_size, report_interval, run_duration,
                        output_interval, disturbance_rate, weathering_rate,
                        dissolution_rate, uplift_interval, plot_interval,
                        friction_coef, fault_x,cell_width, grav_accel,
                        plot_file_name, **kwds)

    def initialize(self, grid_size, report_interval, run_duration,
                   output_interval, disturbance_rate, weathering_rate, 
                   dissolution_rate, uplift_interval, plot_interval,
                   friction_coef, fault_x, cell_width, grav_accel,
                   plot_file_name=None, **kwds):
        """Initialize the grain hill model."""
        self.disturbance_rate = disturbance_rate
        self.weathering_rate = weathering_rate
        self.dissolution_rate = dissolution_rate
        self.uplift_interval = uplift_interval
        self.plot_interval = plot_interval
        self.friction_coef = friction_coef

        self.settling_rate = calculate_settling_rate(cell_width, grav_accel)

        # Call base class init
        super(GrainFacetSimulator, self).initialize(grid_size=grid_size, 
                                          report_interval=report_interval, 
                                          grid_orientation='vertical',
                                          grid_shape='rect',
                                          show_plots=True,
                                          cts_type='oriented_hex',
                                          run_duration=run_duration,
                                          output_interval=output_interval,
                                          plot_every_transition=False)

        # Close top and right edges so as to avoid boundary bug issue
        for edge in (self.grid.nodes_at_right_edge,
                     self.grid.nodes_at_top_edge):
            self.grid.status_at_node[edge] = CLOSED_BOUNDARY

        ns = self.grid.at_node['node_state']
        self.uplifter = LatticeNormalFault(fault_x_intercept=fault_x,
                                           grid=self.grid, 
                                           node_state=ns)

        self.plot_file_name = plot_file_name
        if plot_file_name is not None:
            self.plot_number = 0
            self.plot_to_file()

    def node_state_dictionary(self):
        """
        Create and return dict of node states.

        Overrides base-class method. Here, we simply call on a function in
        the lattice_grain module.
        """
        return lattice_grain_node_states()

    def transition_list(self):
        """
        Make and return list of Transition object.
        """
        xn_list = lattice_grain_transition_list(g=self.settling_rate,
                                                f=self.friction_coef,
                                                motion=self.settling_rate)
        xn_list = self.add_weathering_and_disturbance_transitions(xn_list,
                    self.disturbance_rate, self.weathering_rate,
                    self.dissolution_rate)
        return xn_list

    def add_weathering_and_disturbance_transitions(self, xn_list, d=0.0, w=0.0,
                                                   diss=0.0):
        """
        Add transition rules representing weathering and/or grain disturbance
        to the list, and return the list.

        Parameters
        ----------
        xn_list : list of Transition objects
            List of objects that encode information about the link-state 
            transitions. Normally should first be initialized with lattice-grain
            transition rules, then passed to this function to add rules for
            weathering and disturbance.
        d : float (optional, default=0.0)
            Rate of transition (1/time) from fluid / resting grain pair to
            mobile-grain / fluid pair, representing grain disturbance.
        w : float (optional, default=0.0)
            Rate of transition (1/time) from fluid / rock pair to
            fluid / resting-grain pair, representing weathering.
        diss : float (optional, default=0.0)
            Dissolution: rate of transition from fluid / rock pair to 
            fluid / fluid pair.

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
            xn_list.append( Transition((8,0,0), (7,0,0), w, 'weathering') )
            xn_list.append( Transition((8,0,1), (7,0,1), w, 'weathering') )
            xn_list.append( Transition((8,0,2), (7,0,2), w, 'weathering') )
            xn_list.append( Transition((0,8,0), (0,7,0), w, 'weathering') )
            xn_list.append( Transition((0,8,1), (0,7,1), w, 'weathering') )
            xn_list.append( Transition((0,8,2), (0,7,2), w, 'weathering') )

        # Dissolution rule
        if diss > 0.0:
            xn_list.append( Transition((8,0,0), (0,0,0), diss, 'dissolution') )
            xn_list.append( Transition((8,0,1), (0,0,1), diss, 'dissolution') )
            xn_list.append( Transition((8,0,2), (0,0,2), diss, 'dissolution') )
            xn_list.append( Transition((0,8,0), (0,0,0), diss, 'dissolution') )
            xn_list.append( Transition((0,8,1), (0,0,1), diss, 'dissolution') )
            xn_list.append( Transition((0,8,2), (0,0,2), diss, 'dissolution') )

        if _DEBUG:
            print('')
            print('setup_transition_list(): list has ' + str(len(xn_list))
                  + ' transitions:')
            for t in xn_list:
                print('  From state ' + str(t.from_state) + ' to state '
                      + str(t.to_state) + ' at rate ' + str(t.rate) + 'called'
                      + str(t.name))

        return xn_list

    def initialize_node_state_grid(self):
        """Set up initial node states.

        Examples
        --------
        >>> from grainhill import GrainHill
        >>> gh = GrainHill((5, 7))
        >>> gh.grid.at_node['node_state']        
        array([8, 7, 7, 8, 7, 7, 7, 0, 7, 7, 0, 7, 7, 7, 0, 0, 0, 0, 0, 0, 0, 0,
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
                    nsg[i] = 8

        # Place "wall" particles in the lower-left and lower-right corners
        if self.grid.number_of_node_columns % 2 == 0:
            bottom_right = self.grid.number_of_node_columns - 1
        else:
            bottom_right = self.grid.number_of_node_columns // 2
        nsg[0] = 8  # bottom left
        nsg[bottom_right] = 8

        return nsg

    def run(self):
        """Run the model."""

        # Work out the next times to plot and output
        next_output = self.output_interval
        next_plot = self.plot_interval

        # Next time for a progress report to user
        next_report = self.report_interval

        # And baselevel adjustment
        next_uplift = self.uplift_interval

        current_time = 0.0
        while current_time < self.run_duration:

            # Figure out what time to run to this iteration
            next_pause = min(next_output, next_plot)
            next_pause = min(next_pause, next_uplift)
            next_pause = min(next_pause, self.run_duration)

            # Once in a while, print out simulation and real time to let the user
            # know that the sim is running ok
            current_real_time = time.time()
            if current_real_time >= next_report:
                print('Current sim time' + str(current_time) + '(' + \
                      str(100 * current_time / self.run_duration) + '%)')
                next_report = current_real_time + self.report_interval

            # Run the model forward in time until the next output step
            print('Running to...' + str(next_pause))
            self.ca.run(next_pause, self.ca.node_state)
            current_time = next_pause

            # Handle output to file
            if current_time >= next_output:
                next_output += self.output_interval

            # Handle plotting on display
            if current_time >= next_plot:
                self.ca_plotter.update_plot()
                if self.plot_file_name is not None:
                    self.plot_to_file()
                next_plot += self.plot_interval

            # Handle fault slip
            if current_time >= next_uplift:
                self.uplifter.do_offset(ca=self.ca, current_time=current_time,
                                        rock_state=8)
                next_uplift += self.uplift_interval

    def nodes_in_column(self, col, num_rows, num_cols):
        """Return array of node IDs in given column.
        
        Examples
        --------
        >>> gfs = GrainFacetSimulator((3, 5))
        >>> gfs.nodes_in_column(1, 3, 5)
        array([ 3,  8, 13])
        >>> gfs.nodes_in_column(4, 3, 5)
        array([ 2,  7, 12])
        >>> gfs = GrainFacetSimulator((3, 6))
        >>> gfs.nodes_in_column(3, 3, 6)
        array([ 4, 10, 16])
        >>> gfs.nodes_in_column(4, 3, 6)
        array([ 2,  8, 14])
        """
        base_node = (col // 2) + (col % 2) * ((num_cols + 1) // 2)
        num_nodes = num_rows * num_cols
        return np.arange(base_node, num_nodes, num_cols)

    def get_profile_and_soil_thickness(self):
        """Calculate and return the topographic profile and the regolith
        thickness."""
        nr = self.ca.grid.number_of_node_rows
        nc = self.ca.grid.number_of_node_columns
        data = self.ca.node_state

        elev = np.zeros(nc)
        soil = np.zeros(nc)
        for c in range(nc):
            e = (c%2)/2.0
            s = 0
            r = 0 
            while r<nr and data[c*nr+r]!=0:
                e+=1
                if data[c*nr+r]==7:
                    s+=1
                r+=1
            elev[c] = e
            soil[c] = s
        return elev, soil

    def report_info_for_debug(self, current_time):
        """Print out various bits of data, for testing and debugging."""
        print('\n Current time: ' + str(current_time))
        print('Node state:')
        print(self.ca.node_state)
        for lnk in range(self.grid.number_of_links):
            if self.grid.status_at_link[lnk] == 0:
                print((lnk, self.grid.node_at_link_tail[lnk], 
                       self.grid.node_at_link_head[lnk],
                       self.ca.node_state[self.grid.node_at_link_tail[lnk]], 
                       self.ca.node_state[self.grid.node_at_link_head[lnk]],
                       self.ca.link_state[lnk],self.ca.next_update[lnk],
                       self.ca.next_trn_id[lnk]))
        print('PQ:')
        print(self.ca.priority_queue._queue)
        
    def plot_to_file(self):
        """Plot profile of hill to file."""
        fname = self.plot_file_name + str(self.plot_number).zfill(4) + '.png'
        plot_hill(self.ca.grid, filename=fname)
        self.plot_number += 1


def get_params_from_input_file(filename):
    """Fetch parameter values from input file."""
    from landlab.core import load_params
    
    mpd_params = load_params(filename)

    return mpd_params

def calculate_settling_rate(cell_width, grav_accel):
    """
    Calculate and store gravitational settling rate constant, based on
    given cell size and gravitational acceleration.
    
    Parameters
    ----------
    cell_width : float
        Width of cells, m
    grav_accel : float
        Gravitational acceleration, m/s^2

    Notes
    -----
    Returns settling rate in yr^-1, with the conversion from s to yr 
    calculated using 1 year = 365.25 days.
    
    Examples
    --------
    >>> round(calculate_settling_rate(1.0, 9.8))
    69855725.0
    """
    time_to_settle_one_cell = np.sqrt(2.0 * cell_width / grav_accel)
    return SECONDS_PER_YEAR / time_to_settle_one_cell

def main(params):
    """Initialize model with dict of params then run it."""
    grid_size = (int(params['number_of_node_rows']), 
                 int(params['number_of_node_columns']))
    grain_facet_model = GrainFacetSimulator(grid_size, **params)
    grain_facet_model.run()



if __name__=='__main__':
    """Executes model."""
    try:
        infile = sys.argv[1]
    except IndexError:
        print('Must include input file name on command line')
        sys.exit(1)

    params = get_params_from_input_file(infile)
    main(params)
