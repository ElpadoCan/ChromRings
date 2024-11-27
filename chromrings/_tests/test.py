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

from chromrings import tables_path, core, data_info_json_path
from chromrings.current_analysis import (
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    batch_name
)

NORMALISE_BY_MAX = False
PLOTS = ['Pol I', 'Pol II', 'Pol II']

filename_prefix = (
    f'{batch_name}'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
)

profiles_filename = f'{filename_prefix}_profiles.parquet'
profiles_summary_filename = f'{filename_prefix}_profiles_summary.parquet'

df_summary_path = os.path.join(tables_path, profiles_summary_filename)
df_profiles_path = os.path.join(tables_path, profiles_filename)

df_summary = pd.read_parquet(df_summary_path)
df_profiles = pd.read_parquet(df_profiles_path).reset_index()

