import os
import json

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
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    data_info_json_path, batch_name
)
from chromrings.core import keep_last_point_less_nans

SAVE = True
NORMALISE_BY_MAX = False
CI_METHOD = '95perc_standard_error' # 'min_max'
FED_DATASET = '7_WT_starved_vs_fed_histone' # '1_test_3D_vs_2D'
FED_EXP_FOLDER = 'WT fed-his' # '2D_seg/fed'
STR_EXP_FOLDER = 'WT starved-his' # '2D_seg/str'

filename_prefix_refed = (
    f'4_WT_refed'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

profiles_filename = f'{filename_prefix_refed}_profiles.parquet'

df_profiles_path_refed = os.path.join(tables_path, profiles_filename)
df_profiles_refed = pd.read_parquet(df_profiles_path_refed).reset_index()

filename_prefix_fed = (
    f'{FED_DATASET}'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

profiles_filename = f'{filename_prefix_fed}_profiles.parquet'

df_profiles_path_fed = os.path.join(tables_path, profiles_filename)
df_profiles_fed = pd.read_parquet(df_profiles_path_fed).reset_index()

fed_plot = {
    'exp_folder': FED_EXP_FOLDER,
    'color': 'royalblue',
    'label': 'Fed',
    'df': df_profiles_fed
}
refed_30min_plot = {
    'exp_folder': 'WT-30min refed',
    'color': 'darkturquoise',
    'label': 'Re-fed (30 min)',
    'df': df_profiles_refed
}
refed_5min_plot = {
    'exp_folder': 'WT-5min refed',
    'color': 'deeppink',
    'label': 'Re-fed (5 min)',
    'df': df_profiles_refed
}
str_plot = {
    'exp_folder': STR_EXP_FOLDER,
    'color': 'orangered',
    'label': 'Starved',
    'df': df_profiles_fed
}

fig, ax = plt.subplots(figsize=(6,5))
fig.subplots_adjust(
    left=0.1, bottom=0.1, right=0.95, top=0.95
)
plots = [fed_plot, refed_30min_plot, refed_5min_plot, str_plot]
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