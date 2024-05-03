from collections import defaultdict

import os
import json

import math
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

import scipy.stats

from cellacdc.plot import heatmap

# import diptest

from chromrings import tables_path, figures_path
from chromrings import (
    data_info_json_path, batch_name, utils, USE_ABSOLUTE_DIST, 
    NORMALISE_AVERAGE_PROFILE, NORMALIZE_EVERY_PROFILE, 
    USE_MANUAL_NUCLEOID_CENTERS
)
from chromrings.core import keep_last_point_less_nans

SAVE = True
NORMALISE_BY_MAX = False
CI_METHOD = '95perc_standard_error' # 'min_max'
STAT_TO_PLOT = 'mean' # 'CV', 'skew', 'mean'
PHYSICAL_SIZE_X = 0.1346344

df_profiles, profiles_filename = utils.read_df_profiles(stat_to_plot=STAT_TO_PLOT)
print('*'*60)
answer = input(
    f'Plotting from table: {profiles_filename}\n'
    'Continue ([Y]/N)? '
)
if answer.lower() == 'n':
    exit('Execution stopped')

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
    df_profiles = df_profiles.set_index('dist_perc')
    abs_max = df_profiles.max(axis=None)
    df_profiles = (df_profiles/abs_max).reset_index()
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

if NORMALISE_AVERAGE_PROFILE or NORMALIZE_EVERY_PROFILE:
    y_axis_label = f'Normalised {STAT_TO_PLOT}'
else:
    y_axis_label = f'{STAT_TO_PLOT} [a.u.]'

with open(data_info_json_path, 'r') as json_file:
    data_info = json.load(json_file)

batch_info = data_info[batch_name]
figs = batch_info['figs']
plots = batch_info["plots"]
exp_foldernames = batch_info['experiments']
colors = batch_info['colors']

df_long = (
    pd.melt(df_profiles, id_vars=[('dist_perc', '', '')])
    .rename(columns={
        'variable_0': 'experiment',
        'variable_1': 'Position_n',
        'variable_2': 'ID',
        'value': 'normalised_intensity'
    })
    .set_index(['experiment', 'Position_n', 'ID'])
    .sort_index()
)
df_long.columns = ['dist_perc', 'normalised_intensity']

def statistic(curve_1, curve_2, axis=1):
    return np.mean(curve_1, axis=axis) - np.mean(curve_2, axis=axis)

def cohen_effect_size(curve_1, curve_2):
    mean_1 = np.mean(curve_1, axis=1)
    mean_2 = np.mean(curve_2, axis=1)
    std_1 = np.std(curve_1, axis=1)
    std_2 = np.std(curve_2, axis=1)
    
    std_pooled = np.sqrt((np.square(std_1)+np.square(std_2))/2)
    
    effect_size = (mean_2-mean_1)/std_pooled
    return effect_size
    

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
    nrows = 3
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*5,9))
    fig.subplots_adjust(
        left=0.07, bottom=0.06, right=0.94, top=0.95
    )
    # fig_pt, ax_pt = plt.subplots(1, 2, figsize=(12, 5))
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
    
    data_permut_test = defaultdict(list)
    xx_permut_test = {}
    for col, exp_folder in enumerate(plot_experiments):
        if not exp_folder.startswith(group_name):
            continue
        
        """Heatmap"""
        axis = ax[0, col]
        data = (
            df_long.loc[exp_folder]
            .reset_index()
            .groupby(['Position_n', 'ID'], as_index=False)
            .apply(clip_dist_perc_above_100, include_groups=True)
            .reset_index()
            .fillna(method='ffill')
        )

        # for name, group in data.groupby(['Position_n', 'ID']):
        #     if name == ('Position_9', 'ID_12_mean_radial_profile'):
        #         import pdb; pdb.set_trace()
        
        
        _, _, im = heatmap(
            data,
            x='dist_perc', 
            z='normalised_intensity',
            y_grouping=('Position_n', 'ID'),
            ax=axis,
            num_xticks=5,
            x_labels=x_labels,
            z_min=0,
            z_max=1.0,
            add_x_0_label=add_x_0_label, 
            imshow_kwargs={'aspect': 'auto'}
        )
        # print(exp_folder, len(data.groupby(['Position_n', 'ID'])))
        # print(im.get_array().shape)
        # import pdb; pdb.set_trace()
        axis.set_xlabel(x_label)
        axis.set_title(exp_folder)
        axis.set_yticks([])
        
        # if add_vline_zero:
        #     axis.axvline(0, color='r', ls='--')

        """Line plot"""
        if plots_group[0] == 'all':
            agg_col = 0
        else:
            # agg_col = plots_group.index(exp_folder) // 2 # len(plot_experiments)
            for c, sub_plots in enumerate(plots):
                if sub_plots.find(exp_folder) != -1:
                    agg_col = c
        for key, color in colors.items():
            if exp_folder.find(key) != -1:
                break
        
        # import pdb; pdb.set_trace()
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
        axis.set_ylabel(y_axis_label)
        if add_vline_zero:
            axis.axvline(0, color='r', ls='--')
        
        data_permut_test[agg_col].append(data.values)
        xx_permut_test[agg_col] = xx_plot
    
    for agg_col, data_test in data_permut_test.items():
        if len(data_test) != 2:
            continue
        permutation_result = scipy.stats.permutation_test(
            data_test, 
            statistic=statistic, 
            vectorized=True, 
            axis=1, 
            alternative='two-sided'
        )
        pvalues = permutation_result.pvalue
        effect_sizes = cohen_effect_size(*data_test)
        ax[2, agg_col].plot(xx_plot, -np.log10(pvalues))
        ax[2, agg_col].set_xlabel(x_label)
        ax[2, agg_col].set_ylabel('-$log_{10}(p)$')
        
        ax[2, agg_col+1].plot(xx_plot, np.abs(effect_sizes))
        ax[2, agg_col+1].set_xlabel(x_label)
        ax[2, agg_col+1].set_ylabel('Effect size (Cohen)')
        
        # title = ' vs '.join(plot_experiments)
        # fig_pt.suptitle(title)
        
    
    if SAVE:
        pdf_filename = f'{batch_name}_{group_name}'
        if USE_MANUAL_NUCLEOID_CENTERS:
            pdf_filename = f'{pdf_filename}_with_manual_centroids'
        pdf_filepath = os.path.join(figures_path, f'{pdf_filename}.pdf')
        fig.savefig(pdf_filepath)

plt.show()