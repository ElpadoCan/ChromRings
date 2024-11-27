import os
import json

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from cellacdc.plot import heatmap


from chromrings import tables_path, figures_path, data_info_json_path
from chromrings.current_analysis import (
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    batch_name
)
from chromrings.core import keep_last_point_less_nans

SAVE = False
NORMALISE_BY_MAX = False
CI_METHOD = '95perc_standard_error' # 'min_max'

filename_prefix_refed = (
    f'5_WT_starved_DNA_vs_histone'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

profiles_filename = f'{filename_prefix_refed}_profiles.parquet'

df_profiles_path_str = os.path.join(tables_path, profiles_filename)
df_profiles_str = pd.read_parquet(df_profiles_path_str).reset_index()

filename_prefix_fed = (
    f'1_test_3D_vs_2D'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

profiles_filename = f'{filename_prefix_fed}_profiles.parquet'

df_profiles_path_fed = os.path.join(tables_path, profiles_filename)
df_profiles_fed = pd.read_parquet(df_profiles_path_fed).reset_index()

fed_plot = {
    'exp_folder': '2D_seg/fed',
    'color': 'royalblue',
    'label': 'Fed',
    'df': df_profiles_fed
}
str_dna_plot = {
    'exp_folder': 'WT starved-DNA',
    'color': 'deeppink',
    'label': 'Starved (DNA)',
    'df': df_profiles_str
}
str_hist_plot = {
    'exp_folder': 'WT starved-histone',
    'color': 'orangered',
    'label': 'Starved (Histone)',
    'df': df_profiles_str
}

fig, ax = plt.subplots(figsize=(6,5))
fig.subplots_adjust(
    left=0.1, bottom=0.1, right=0.95, top=0.95
)
plots = [fed_plot, str_dna_plot, str_hist_plot]
for plot_kwargs in plots:
    df = plot_kwargs['df']
    exp_folder = plot_kwargs['exp_folder']
    color = plot_kwargs['color']
    label = plot_kwargs['label']
        
    data = df.set_index('dist_perc')[exp_folder]
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

    # Plot at center of bin --> move left by 2.5
    ax.plot(
        data_agg.index-2.5, data_agg.values, 
        color=color, 
        label=label,
    )
    ax.fill_between(
        data_agg.index-2.5, data_y_low,data_y_high, 
        color=color,
        alpha=0.3,
    )
ax.legend()
ax.set_xticks(np.arange(0,101,20))
ax.set_ylim((0,1.05))
ax.set_xlabel('Distance from nucleus center of mass (%)')
ax.set_ylabel('Normalised mean intensity')

if SAVE:
    fig.savefig(os.path.join(figures_path, f'4_WT_str_refed_fed.pdf'))

plt.show()