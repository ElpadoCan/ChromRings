import os

from collections import defaultdict

import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from chromrings import tables_path, data_path
from chromrings import core, utils

# '13_nucleolus_nucleus_profile'
MUSCLES_NUCLEOLUS_DATA = '27_muscles_resol_limit'
EXPERIMENT = None # 'Starved_with_spotmax_3D_seg'

DEBUG = False

muscles_data_path = os.path.join(data_path, '27_muscles_resol_limit')

# Read nucleolus edge profiles
df_profiles_ne, _ = utils.read_df_profiles(
    batch_name=MUSCLES_NUCLEOLUS_DATA, 
    stat_to_plot='mean', 
    load_absolute_dist=True
)
df_profiles_ne_str = (
    df_profiles_ne
    .dropna()
    .set_index('dist_perc')
)
if EXPERIMENT is not None:
    df_profiles_ne_str = df_profiles_ne_str[[EXPERIMENT]]

# Fit peaks to starved where we have nucleolus edge distance
_, df_coeffs_ne = core.fit_profiles(
    df_profiles_ne_str, inspect=False, show_pbar=True, 
    init_guess_peaks_loc=[2, 7], A_init=0.5, 
    peak_center_max_range=1.5
)

# Read L1s starved fit coeffs (from chromrings.plot.fit_adults_vs_L1)
df_fit_path = os.path.join(
    tables_path, 'adults_vs_L1s_starved_fit_coeffs.parquet'
)
df_fit = pd.read_parquet(df_fit_path).reset_index()
df_fit_L1s = df_fit[df_fit['stage'] == 'L1s'].copy()

# Get distance outer peak to nucleus edge
df_fit_L1s_outer_peak = df_fit_L1s[df_fit_L1s['peak_idx'] == 'peak_1'].copy()
mean_outer_dist_perc = (100 - df_fit_L1s_outer_peak['xc_fit'].mean())/100

# Get distance inner peak to nucleolus edge
df_coeffs_ne = df_coeffs_ne.reset_index()
df_fit_ne_inner_peak = df_coeffs_ne[df_coeffs_ne['peak_idx'] == 'peak_0'].copy()
mean_inner_dist_pxl = df_fit_ne_inner_peak['xc_fit'].mean()

# See https://en.wikipedia.org/wiki/Airy_disk#Approximation_using_a_Gaussian_profile
# where it says:
# using the approximation above shows that the Gaussian waist of the Gaussian 
# approximation to the Airy disk is about two-third the Airy disk radius...
average_sigma = df_coeffs_ne['sigma_fit'].mean()
airy_radius = (1.22/0.84)*average_sigma
df_coeffs_ne['airy_radius'] = (1.22/0.84)*df_coeffs_ne['sigma_fit']

dfs_dist = {}
for root, dirst, files in os.walk(muscles_data_path):
    if not root.endswith('Images'):
        continue
    
    nucleolus_lab = None
    nucleus_lab = None
    for file in files:
        if file.endswith('segm.npz'):
            nucleus_lab = np.load(os.path.join(root, file))['arr_0']
        elif file.endswith('segm_old_spotmax.npz'):
            nucleolus_lab = np.load(os.path.join(root, file))['arr_0']
    
    df_dist = core.radial_distances_nucleolus_nucleus(
        nucleolus_lab, nucleus_lab, debug=DEBUG
    )
    
    df_dist['outer_peak_to_nucleus_edge_dist_pxl'] = (
        df_dist['center_to_edge_nucleus_distance']*mean_outer_dist_perc
    )
    
    df_dist['peak_to_peak_distance'] = (
        df_dist['nucleolus_to_nucleus_distance'] 
        - df_dist['outer_peak_to_nucleus_edge_dist_pxl'] 
    )
    df_dist['peak_to_peak_distance'] = (
        df_dist['nucleolus_to_nucleus_distance'] - mean_inner_dist_pxl
    )
    
    dfs_dist[root] = df_dist

df_dist = pd.concat(
    dfs_dist.values(), 
    keys=dfs_dist.keys(), 
    names=['images_path']
).reset_index()

ycol = 'Distance [pixel]'
df_dist[ycol] = df_dist['nucleolus_to_nucleus_distance']
df_coeffs_ne[ycol] = df_coeffs_ne['airy_radius']

df_peak_to_peak = df_dist[['peak_to_peak_distance']].copy()
df_peak_to_peak[ycol] = df_peak_to_peak['peak_to_peak_distance']

df_data = pd.concat(
    [df_coeffs_ne[[ycol]], df_dist[[ycol]], df_peak_to_peak[[ycol]]], 
    keys=[
        'Resolvable distance', 
        'Nucleolus to nucleus distance', 
        'Peak to peak distance'
    ], 
    names=['Category']
)

sigma_resolution = average_sigma
x0_inner = mean_inner_dist_pxl
x0_outer = x0_inner + df_dist['peak_to_peak_distance'].mean()
coeffs = np.array([x0_inner, average_sigma, 1, x0_outer, average_sigma, 1])
model = core.PeaksModel(n_peaks=2)
X = np.linspace(0, x0_outer+mean_inner_dist_pxl, 200)
simulated_data = model.model(X, coeffs, 0)

fig, ax = plt.subplots(1, 2)

ax[1].plot(X, simulated_data)

sns.boxplot(
    data=df_data, 
    x='Category', 
    y=ycol,
    ax=ax[0]
)
plt.show()

import pdb; pdb.set_trace()

    