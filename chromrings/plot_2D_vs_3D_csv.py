import os

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# import diptest

from chromrings import tables_path, core
from chromrings import (
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
)

import seaborn as sns

NORMALISE_BY_MAX = False

exp_folder = ('1_test_3D_vs_2D_01-02-2023', '2D_seg') 
exp_foldername = '_'.join(exp_folder)
filename_prefix = (
    f'{exp_foldername}'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

filename_prefix_3D = filename_prefix.replace('_2D_seg', '_3D_seg')

"""Average profile starved vs fed"""
df_profiles_fed_path_2D = os.path.join(tables_path, f'{filename_prefix}_profiles_fed.csv')
df_profiles_fed_2D = pd.read_csv(df_profiles_fed_path_2D, index_col='dist_perc')
if NORMALISE_BY_MAX:
    df_profiles_fed_2D = df_profiles_fed_2D / df_profiles_fed_2D.max()
df_fed_average_2D = df_profiles_fed_2D.mean(axis=1).dropna()

df_profiles_starved_path_2D = os.path.join(tables_path, f'{filename_prefix}_profiles_starved.csv')
df_profiles_starved_2D = pd.read_csv(df_profiles_starved_path_2D, index_col='dist_perc')
if NORMALISE_BY_MAX:
    df_profiles_starved_2D = df_profiles_starved_2D / df_profiles_starved_2D.max()
df_starved_average_2D = df_profiles_starved_2D.mean(axis=1).dropna()


df_profiles_fed_path_3D = os.path.join(tables_path, f'{filename_prefix_3D}_profiles_fed.csv')
df_profiles_fed_3D = pd.read_csv(df_profiles_fed_path_3D, index_col='dist_perc')
if NORMALISE_BY_MAX:
    df_profiles_fed_3D = df_profiles_fed_3D / df_profiles_fed_3D.max()
df_fed_average_3D = df_profiles_fed_3D.mean(axis=1).dropna()

df_profiles_starved_path_3D = os.path.join(tables_path, f'{filename_prefix_3D}_profiles_starved.csv')
df_profiles_starved_3D = pd.read_csv(df_profiles_starved_path_3D, index_col='dist_perc')
if NORMALISE_BY_MAX:
    df_profiles_starved_3D = df_profiles_starved_3D / df_profiles_starved_3D.max()
df_starved_average_3D = df_profiles_starved_3D.mean(axis=1).dropna()

ndims = ['2D', '3D']

fig, ax = plt.subplots()
colors = sns.color_palette(n_colors=2)
for ndim in ndims:
    if ndim == '2D':
        df_fed_average = df_fed_average_2D
        df_starved_average = df_starved_average_2D
    else:
        df_fed_average = df_fed_average_3D
        df_starved_average = df_starved_average_3D

    ax.plot(df_fed_average.index , df_fed_average.values, color=colors[0])
    ax.plot(df_starved_average.index , df_starved_average.values, color=colors[1])

    """Find peaks"""
    find_peaks_kwargs = {}
    peaks_fed, props_fed = find_peaks(df_fed_average.values, **find_peaks_kwargs)
    xx_peaks_fed = df_fed_average.index[peaks_fed]
    yy_peaks_fed = df_fed_average.values[peaks_fed]

    ax.scatter(xx_peaks_fed, yy_peaks_fed, s=80, facecolors='none', edgecolors=colors[0])

    peaks_str, props_str = find_peaks(df_starved_average.values, **find_peaks_kwargs)
    xx_peaks_str = df_starved_average.index[peaks_str]
    yy_peaks_str = df_starved_average.values[peaks_str]

    ax.scatter(xx_peaks_str, yy_peaks_str, s=80, facecolors='none', edgecolors=colors[1])

ax.set_title('Average of all profiles')

legend_handles = []
for s, label in enumerate(['fed', 'starved']):
    legend_handles.append(
        mpatches.Patch(color=colors[s], label=label)
    )

fig.legend(handles=legend_handles, loc='center right')

fig.suptitle(f'{filename_prefix}')

plt.show()