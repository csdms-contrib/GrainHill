# -*- coding: utf-8 -*-
"""
Simple driver for GrainHill model

"""

#import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from grainhill import GrainFacet, CosmogenicIrradiator
from landlab import imshow_grid


GRAV_ACCEL = 9.8
SEC_PER_YEAR = 3600 * 24 * 365.25

delta = 0.75  # cell width in meters
#source_dir = '/Users/gtucker/Dev/MountainFrontModel/mountain_front_model'

#start_dir = os.getcwd()
#os.chdir(source_dir)
#grainhill = reload(grainhill)
#os.chdir(start_dir)


def plot_hill(grid, filename=None, array=None):
    """Generate a plot of the modeled hillslope."""

    # Set color map
    rock = '#5F594D'
    sed = '#A4874B'
    sky = '#D0E4F2'
    mob = '#D98859'
    clist = [sky, mob, mob, mob, mob, mob, mob, sed, rock]
    my_cmap = mpl.colors.ListedColormap(clist)

    if array is None:
        array = grid.at_node['node_state']

    # Generate the plot
    ax = grid.hexplot(array, color_map=my_cmap)
    #plt.axis('off')
    ax.set_aspect('equal')

    # If applicable, save to file. Otherwise display the figure.
    # (Note: the latter option freezes execution until user dismisses window)
    if filename is not None:
        plt.savefig(filename, bbox_inches='tight')
        plt.clf()
        print('Figure saved to ' + filename)
    else:
        plt.show()





# Dictionary for parameters
params = {}

num_cols = 51  # 51
params['number_of_node_columns'] = num_cols
num_rows = 71  # 31
params['number_of_node_rows'] = num_rows
params['disturbance_rate'] = 0.001
params['uplift_interval'] = 100.0
params['weathering_rate'] = 0.01
params['run_duration'] = 5000.0  # 3000.
params['uplift_duration'] = params['run_duration']
params['show_plots'] = True
params['plot_interval'] = 50.0
params['output_interval'] = 10000001.0
params['report_interval'] = 20.0
params['settling_rate'] = np.sqrt(delta * GRAV_ACCEL / 2.0) * SEC_PER_YEAR
params['friction_coef'] = 1.0
params['fault_x'] = 5.0
params['rock_state_for_uplift'] = 8
params['opt_rock_collapse'] = 1.0
params['opt_track_grains'] = True

# Cosmo parameters
cosmo_interval = 50.0
cosmo_prod_rate = 1.0
cosmo_decay_depth = 0.6

next_cosmo = cosmo_interval

# Create a field for cosmo concentration
params['prop_reset_value'] = 0.0
params['prop_data'] = 'cosmogenic_nuclide__concentration'

# instantiate a GrainFacet model
gh = GrainFacet((num_rows, num_cols), **params)

# instantiate a Cosmo handler
ci = CosmogenicIrradiator(gh, cosmo_prod_rate, cosmo_decay_depth)


#run the model
current_time = 0.0
while current_time < params['run_duration']:
    run_to = min(next_cosmo, params['run_duration'])
    print('current time ' + str(current_time))
    print('running to ' + str(run_to))
    gh.run(to=run_to)
    ci.add_cosmos(run_to - current_time, delta)
    next_cosmo += cosmo_interval
    current_time = run_to


plot_hill(gh.grid, 'grain_hill_test_cosmo_ns.png')
gh.grid.hexplot(ci.cosmo[gh.ca.propid])
plt.savefig('grain_hill_test_cosmo_co.png', bbox_inches='tight')
