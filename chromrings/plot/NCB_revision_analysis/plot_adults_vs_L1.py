import os

from tqdm import tqdm

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from chromrings import tables_path, figures_path
from chromrings import utils, core

TABLE_FILENAME = 'adults_vs_L1s_starved'

coeffs_filename = f'{TABLE_FILENAME}_fit_coeffs.parquet'
coeffs_filepath = os.path.join(tables_path, coeffs_filename)
df_coeffs_fit = (
    pd.read_parquet(coeffs_filepath)
    .reset_index()
    .set_index('peak_idx')
)

df_coeffs_peak_0 = df_coeffs_fit.loc['peak_0'].set_index('stage')
df_coeffs_peak_1 = df_coeffs_fit.loc['peak_1'].set_index('stage')


df_A_ratio_peaks = (
    (df_coeffs_peak_1.A_fit/df_coeffs_peak_0.A_fit)
    .dropna()
    .reset_index()
    .rename(columns={'A_fit': 'Ratio amplitudes second/first peak'})
)
# import pdb; pdb.set_trace()
sns.boxplot(
    x='stage', 
    y='Ratio amplitudes second/first peak', 
    data=df_A_ratio_peaks
)

plt.show()
