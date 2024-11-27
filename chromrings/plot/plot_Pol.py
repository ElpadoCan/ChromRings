import os
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from cellacdc.plot import heatmap

# import diptest

from chromrings import tables_path, figures_path, data_info_json_path, utils
from chromrings.current_analysis import USE_MANUAL_NUCLEOID_CENTERS

SAVE = True
NORMALISE_BY_MAX = False
PLOTS = ['Pol I-', 'Pol II-', 'Pol III-']
CI_METHOD = '95perc_standard_error' # 'min_max'
batch_name = '2_Pol_I_II_III'

STAT_TO_PLOT = 'mean' # 'CV', 'skew', 'mean'

df_profiles, profiles_filename = utils.read_df_profiles(
    stat_to_plot=STAT_TO_PLOT, batch_name=batch_name
)
print('*'*60)
answer = input(
    f'Plotting from table: {profiles_filename}\n'
    'Continue ([Y]/N)? '
)
if answer.lower() == 'n':
    exit('Execution stopped')

with open(data_info_json_path, 'r') as json_file:
    data_info = json.load(json_file)

batch_info = data_info[batch_name]
exp_foldernames = batch_info['experiments']

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

def clip_dist_perc_above_100(group):
    if len(group) == 21:
        clipped = group[group['dist_perc'] < 105]
    else:
        clipped = group.copy()
        clipped.loc[clipped['dist_perc'] == 105, 'dist_perc'] = 100
    return clipped[['dist_perc', 'normalised_intensity']]

for PLOT in PLOTS:
    plot_experiments = [
        exp_folder for exp_folder in exp_foldernames 
        if exp_folder.startswith(PLOT)
    ]
    ncols = len(plot_experiments)
    nrows = 2
    fig, ax = plt.subplots(nrows, ncols, figsize=(ncols*6,9))
    fig.subplots_adjust(
        left=0.06, bottom=0.05, right=0.95, top=0.95
    )
    if ncols == 1:
        ax = ax[:, np.newaxis]
    for col, exp_folder in enumerate(plot_experiments):
        if not exp_folder.startswith(PLOT):
            continue
        
        """Heatmap"""
        axis = ax[0, col]
        data = (
            df_long.loc[exp_folder]
            .reset_index()
            .groupby(['Position_n', 'ID'])
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
            x_labels=np.arange(0,101,20),
            z_min=0,
            z_max=1.0
        )
        
        axis.set_xlabel('Distance from nucleus center of mass (%)')
        axis.set_title(exp_folder)
        axis.set_yticks([])

        """Line plot"""
        axis = ax[1, col]
        
        data = df_profiles.set_index('dist_perc')[exp_folder]
        data = data[data.index < 105]
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
        
        # Plot at center of bin --> move left by 2.5
        axis.plot(
            data_agg.index-2.5, data_agg.values, 
            color='darkturquoise'
        )
        axis.fill_between(
            data_agg.index-2.5, data_y_low,data_y_high, 
            color='darkturquoise',
            alpha=0.3
        )
        axis.set_xticks(np.arange(0,101,20))
        axis.set_ylim((0,1.05))
        axis.set_xlabel('Distance from nucleus center of mass (%)')
        axis.set_ylabel('Normalised mean intensity')
    
    if SAVE:
        pdf_filename = f'{batch_name}_{PLOT}'
        if USE_MANUAL_NUCLEOID_CENTERS:
            pdf_filename = f'{pdf_filename}_with_manual_centroids'
        fig.savefig(os.path.join(figures_path, f'{pdf_filename}.pdf'))

plt.show()