import os
import json

import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from acdctools.plot import heatmap

# import diptest

from chromrings import tables_path, figures_path
from chromrings import (
    data_info_json_path, batch_name, utils, USE_ABSOLUTE_DIST
)
from chromrings.core import keep_last_point_less_nans

SAVE = False
NORMALISE_BY_MAX = False
CI_METHOD = '95perc_standard_error' # 'min_max'
STAT_TO_PLOT = 'mean' # 'CV', 'skew', 'mean'
PHYSICAL_SIZE_X = 0.1346344

df_profiles, profiles_filename = utils.read_df_profiles(stat_to_plot=STAT_TO_PLOT)
print(f'Using table file "{profiles_filename}"')

if (df_profiles['dist_perc'] < 0).any():
    x_labels = np.arange(df_profiles['dist_perc'].min(),101,20)
    add_vline_zero = True
    add_x_0_label = False
    x_label = 'Distance from nucleolus edge'
else:
    x_labels = np.arange(0,101,20)
    add_vline_zero = False
    add_x_0_label = True
    x_label = 'Distance from nucleus center of mass'

if USE_ABSOLUTE_DIST:
    df_profiles['dist'] = df_profiles['dist_perc'] * PHYSICAL_SIZE_X
    min_dist = df_profiles['dist'].min()
    max_dist = df_profiles['dist'].max()
    num_labels = 10
    start = math.floor(min_dist)
    stop = math.ceil(max_dist)
    xrange = stop-start
    left_range = abs(start)
    right_range = stop
    ratio = right_range/xrange
    right_num = int(num_labels*ratio)
    left_num = num_labels - right_num
    left_space = np.linspace(start, 0, left_num)
    if left_space[-1] == 0:
        left_space = left_space[:-1]
    right_space = np.linspace(0, stop, right_num)
    x_dist_labels = np.round(np.concatenate((left_space, right_space)), 2)
    x_label = f'{x_label} [$\mu$m]'
else:
    x_label = f'{x_label} [%]'
    x_dist_labels = x_labels
    df_profiles['dist'] = df_profiles['dist_perc'] 

with open(data_info_json_path, 'r') as json_file:
    data_info = json.load(json_file)

batch_info = data_info[batch_name]
figs = batch_info['figs']
plots = batch_info["plots"]
exp_foldernames = batch_info['experiments']
colors = batch_info['colors']

df_long = (
    pd.melt(df_profiles, id_vars=['dist_perc'])
    .rename(columns={
        'variable_0': 'experiment',
        'variable_1': 'Position_n',
        'variable_2': 'ID',
        'value': 'normalised_intensity'
    })
    .set_index(['experiment', 'Position_n', 'ID'])
    .sort_index()
)

# import pdb; pdb.set_trace()

def clip_dist_perc_above_100(group):
    if len(group) == 21:
        clipped = group[group['dist_perc'] < 105]
    else:
        clipped = group.copy()
        clipped.loc[clipped['dist_perc'] == 105, 'dist_perc'] = 100
    return clipped[['Position_n', 'ID', 'dist_perc', 'normalised_intensity']]

for group_name in figs:
    plot_experiments = [
        exp_folder for exp_folder in exp_foldernames 
        if exp_folder.startswith(group_name)
    ]
    ncols = len(plot_experiments)
    nrows = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*6,9))
    fig.subplots_adjust(
        left=0.06, bottom=0.06, right=0.95, top=0.95
    )
    plots_group = [plot for plot in plots if plot.startswith(group_name)]
    plots_pairs = plots_group.copy()
    if len(plots_group) > 1:
        # Split pairs
        flattened_plots = []
        for plots_group_sub in plots_group:
            flattened_plots.extend(plots_group_sub.split(';;'))
        plots_group = flattened_plots
    else:
        plots_group = plots_group[0].split(';;')
    if ncols == 1:
        ax = ax[:, np.newaxis]
    for col, exp_folder in enumerate(plot_experiments):
        if not exp_folder.startswith(group_name):
            continue
        
        """Heatmap"""
        axis = ax[0, col]
        data = (
            df_long.loc[exp_folder]
            .reset_index()
            .groupby(['Position_n', 'ID'], as_index=False)
            .apply(clip_dist_perc_above_100)
            .reset_index()
            .fillna(method='ffill')
        )

        # for name, group in data.groupby(['Position_n', 'ID']):
        #     if name == ('Position_9', 'ID_12_mean_radial_profile'):
        #         import pdb; pdb.set_trace()
        
        heatmap(
            data,
            x='dist_perc', 
            z='normalised_intensity',
            y_grouping=('Position_n', 'ID'),
            ax=axis,
            num_xticks=5,
            x_labels=x_labels,
            z_min=0,
            z_max=1.0,
            add_x_0_label=add_x_0_label
        )
        
        axis.set_xlabel(x_label)
        axis.set_title(exp_folder)
        axis.set_yticks([])
        
        # if add_vline_zero:
        #     axis.axvline(0, color='r', ls='--')

        """Line plot"""
        agg_col = col // len(plot_experiments)
        for key, color in colors.items():
            if exp_folder.find(key) != -1:
                break

        axis = ax[1, agg_col]
        
        data = df_profiles.set_index('dist')[exp_folder]
        data = keep_last_point_less_nans(data)
        data_agg = data.mean(axis=1)
        if CI_METHOD =='min_max':
            data_y_low = data.min(axis=1).values
            data_y_high = data.max(axis=1).values
        elif CI_METHOD == '95perc_standard_error':
            stds = data.std(axis=1)
            num_obs = len(data.columns)
            std_errs = stds/(num_obs**(1/2))
            data_y_low = (data_agg - 2*std_errs).values
            data_y_high = (data_agg + 2*std_errs).values
        
        linestyle = '-'
        if exp_folder.find('3D_seg') != -1:
            linestyle = '--'
        
        if USE_ABSOLUTE_DIST:
            xx_plot = data_agg.index
        else:
            # Plot at center of bin --> move left by 2.5
            xx_plot = data_agg.index-2.5
        
        axis.plot(
            xx_plot, data_agg.values, 
            color=color, 
            label=exp_folder,
            linestyle=linestyle
        )
        axis.fill_between(
            xx_plot, data_y_low,data_y_high, 
            color=color,
            alpha=0.3,
            linestyle=linestyle
        )
        axis.legend()
        axis.set_xticks(x_dist_labels)
        axis.set_ylim((0,1.05))
        axis.set_xlabel(x_label)
        axis.set_ylabel(f'Normalised {STAT_TO_PLOT}')
        if add_vline_zero:
            axis.axvline(0, color='r', ls='--')
    
    if SAVE:
        fig.savefig(os.path.join(figures_path, f'{batch_name}_{group_name}.pdf'))

plt.show()