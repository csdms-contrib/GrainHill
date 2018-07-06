# -*- coding: utf-8 -*-
"""
Simple driver for GrainHill model

GT May 2018
"""

#import os
import numpy as np
from grainhill import GrainHill, BlockHill, plot_hill
from landlab.core import load_params
import sys
import os


SEC_PER_YEAR = 3600 * 24 * 365.25


# Function to set up a nicer colormap than the defaults, for Block option
def get_block_hill_colormap():
    """Create and return a listed colormap."""
    import matplotlib as mpl

    rock = '#5F594D'
    sed = '#A4874B'
    sky = '#D0E4F2'
    mob = '#D98859'
    block = '#660000'
    rock = '#000000'
    clist = [sky, mob, mob, mob, mob, mob, mob, sed, rock, block]
    return mpl.colors.ListedColormap(clist)


def plot_blocky_hill(grid, filename=None, array=None):
    """Generate a plot of the modeled hillslope."""
    import matplotlib.pyplot as plt

    # Set color map
    cmap = get_block_hill_colormap()

    if array is None:
        array = grid.at_node['node_state']

    # Generate the plot
    ax = grid.hexplot(array, color_map=cmap)
    ax.set_aspect('equal')

    # If applicable, save to file. Otherwise display the figure.
    # (Note: the latter option freezes execution until user dismisses window)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        print('Figure saved to ' + filename)
    else:
        plt.show()



class GrainHillDriver(object):
    """Driver for the GrainHill model: handles initialization, run, cleanup."""

    def __init__(self, input_file=None):
        """Construct a GrainHillDriver; if passed a file name, initialize."""

        if input_file is not None:
            self.initialize(input_file)
        else:
            self.initialized = False

    def initialize(self, input_file_name):
        """Initialize by reading parameters from input file."""
        params = load_params(input_file_name)
        settling_rate = np.sqrt(params['cell_size'] * params['grav_accel']
                                / 2.0) * SEC_PER_YEAR
        self.run_duration = params['run_duration']
        self.output_name = params['output_name']
        if 'uplift_duration' in params:
            self.uplift_duration = params['uplift_duration']
        else:
            self.uplift_duration = self.run_duration

        if 'include_blocks' in params and params['include_blocks']:
            if 'y0_top' in params:
                y0_top = params['y0_top']
            else:
                y0_top = 0.0
            if 'layer_left_x' in params:
                layer_left_x = params['layer_left_x']
            else:
                layer_left_x = 0.0
            self.model = BlockHill((params['number_of_node_rows'],
                             params['number_of_node_columns']),
                            report_interval=params['report_interval'],
                            run_duration=self.run_duration,
                            output_interval=params['output_interval'],
                            settling_rate=settling_rate,
                            disturbance_rate=params['disturbance_rate'],
                            weathering_rate=params['weathering_rate'],
                            uplift_interval=params['uplift_interval'],
                            plot_interval=params['plot_interval'],
                            friction_coef=params['friction_coef'],
                            rock_state_for_uplift=params['rock_state_for_uplift'],
                            opt_rock_collapse=params['opt_rock_collapse'],
                            show_plots=params['show_plots'],
                            initial_state_grid=None,
                            opt_track_grains=params['opt_track_grains'],
                            block_layer_dip_angle=params['block_layer_dip_angle'],
                            block_layer_thickness=params['block_layer_thickness'],
                            y0_top=y0_top,
                            layer_left_x=layer_left_x,
                            cmap=get_block_hill_colormap())
            self.cmap = get_block_hill_colormap()  # use special colormap
        else:
            self.model = GrainHill((params['number_of_node_rows'],
                             params['number_of_node_columns']),
                            report_interval=params['report_interval'],
                            run_duration=self.run_duration,
                            output_interval=params['output_interval'],
                            settling_rate=settling_rate,
                            disturbance_rate=params['disturbance_rate'],
                            weathering_rate=params['weathering_rate'],
                            uplift_interval=params['uplift_interval'],
                            plot_interval=params['plot_interval'],
                            friction_coef=params['friction_coef'],
                            rock_state_for_uplift=params['rock_state_for_uplift'],
                            opt_rock_collapse=params['opt_rock_collapse'],
                            show_plots=params['show_plots'],
                            initial_state_grid=None,
                            opt_track_grains=params['opt_track_grains'])
            self.cmap = None  # use default GrainHill colormap

        if params['plot_to_file']:
            self.file_plot_interval = params['plot_interval']
        else:
            self.file_plot_interval = self.run_duration
        self.plot_number = 0

        self.find_or_create_output_folder()

        self.initialized = True

    def find_or_create_output_folder(self):
        """Create folder with output_name if it doesn't already exist."""
        if not self.output_name in os.listdir('.'):
            os.makedirs('./' + self.output_name)

    def plot_to_file(self):
        """Plot current hillslope to file."""
        fname = (self.output_name + '/' + self.output_name
                + str(self.plot_number).zfill(4) + '.png')
        plot_hill(self.model.grid, filename=fname, cmap=self.cmap)
        self.plot_number += 1

    def update_until(self, time):
        """Run model up to given time."""
        print('running to ' + str(time))
        self.model.run(to=time)

    def run(self):
        """Run model from start to finish."""
        if not self.initialized:
            print('Must call initialize() before run()')
            raise Exception

        self.plot_to_file()
        next_file_plot = self.file_plot_interval
        uplift_change_time = self.uplift_duration
        while self.model.current_time < self.run_duration:
            next_pause = min(next_file_plot, self.run_duration)
            next_pause = min(next_pause, uplift_change_time)
            if self.model.current_time >= uplift_change_time:
                self.model.next_uplift = self.run_duration
                uplift_change_time = self.run_duration
            self.update_until(next_pause)
            if self.model.current_time >= next_file_plot:
                self.plot_to_file()
                next_file_plot += self.file_plot_interval

    def finalize(self):
        pass



if __name__ == '__main__':

    try:
        filename = sys.argv[1]
    except:
        print('Please specify a file name as the first argument')
        raise

    gh = GrainHillDriver()
    gh.initialize(filename)
    gh.run()
    gh.finalize()
