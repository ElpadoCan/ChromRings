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
    utils
)
from chromrings.core import keep_last_point_less_nans

SAVE = True
NORMALISE_BY_MAX = False
CI_METHOD = '95perc_standard_error' # 'min_max'
STAT_TO_PLOT = 'mean' # 'CV', 'skew', 'mean'
STR_DATASET = '5_WT_starved_DNA_vs_histone' # '1_test_3D_vs_2D'
STR_EXP_FOLDER = 'WT starved-DNA' # '2D_seg/str'
POL_I_DATASET = '2_Pol_I_II_III'
POL_I_EXP_FOLDER = 'Pol I-auxin 3hrs' # '2D_seg/fed'

PDF_FILENAME = '5_2_WT_starved_DNA_vs_Pol_I_degraded_fed.pdf'

filename_prefix_str = (
    f'{STR_DATASET}'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

df_profiles_str, _ = utils.read_df_profiles(
    stat_to_plot=STAT_TO_PLOT, batch_name=STR_DATASET
)


# filename_prefix_fed = (
#     f'{POL_I_DATASET}'
#     f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
#     f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
#     f'_norm_how_{NORMALISE_HOW}'
# )

df_profiles_pol_i, _ = utils.read_df_profiles(
    stat_to_plot=STAT_TO_PLOT, batch_name=POL_I_DATASET
)

fed_plot = {
    'exp_folder': STR_EXP_FOLDER,
    'color': 'orangered',
    'label': 'Wt starved (DNA)',
    'df': df_profiles_str
}
pol_i_plot = {
    'exp_folder': POL_I_EXP_FOLDER,
    'color': 'deeppink',
    'label': 'Pol I degraded fed',
    'df': df_profiles_pol_i
}

fig, ax = plt.subplots(figsize=(6,5))
fig.subplots_adjust(
    left=0.1, bottom=0.1, right=0.95, top=0.95
)
plots = [fed_plot, pol_i_plot]
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
    fig.savefig(os.path.join(figures_path, PDF_FILENAME))

plt.show()