import os

from tqdm import tqdm

import pandas as pd

from chromrings import tables_path, figures_path
from chromrings import utils, core

SAVE = True
STAT_TO_PLOT = 'mean' # 'CV', 'skew', 'mean'

DATASET_INFO = {
    'adults': {
        '12_adults': ["adults-starved-his"]
    },
    'L1s': {
        '1_test_3D_vs_2D': ["2D_seg/str"]
    }
}

TABLE_FILENAME = 'adults_vs_L1s_starved'


pbar = tqdm(total=len(DATASET_INFO), desc='Total', ncols=100, leave=True)
keys = []
dfs_profiles_fit = []
dfs_fit_coeffs = []
for category, datasets in DATASET_INFO.items():
    for dataset, experiments in datasets.items():
        df_profiles, profiles_filename = utils.read_df_profiles(
            batch_name=dataset, stat_to_plot=STAT_TO_PLOT
        )
        df_profiles = (
            df_profiles
            .dropna()
            .set_index('dist_perc')[experiments]
        )

        df_profiles_fit, df_coeffs = core.fit_profiles(
            df_profiles, inspect=False, show_pbar=True, 
            init_guess_peaks_loc=[30, 90]
        )
        keys.append((profiles_filename, category, dataset))
        dfs_profiles_fit.append(df_profiles_fit)
        dfs_fit_coeffs.append(df_coeffs)
    
    pbar.update()

pbar.close()

if SAVE:
    df_coeffs_fit = pd.concat(
        dfs_fit_coeffs, 
        keys=keys, 
        names=['table_filename', 'stage', 'dataset']
    )

    coeffs_filename = f'{TABLE_FILENAME}_fit_coeffs.parquet'
    coeffs_filepath = os.path.join(tables_path, coeffs_filename)
    df_coeffs_fit.to_parquet(coeffs_filepath)
        