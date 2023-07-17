import os

import pandas as pd

from chromrings import data_path

dataset_name = '13_nucleolus_nucleus_profile'

nucleolus_fed = 'AllPos_acdc_output_nucleolus_tracked_fed.csv'
nucleolus_str = 'AllPos_acdc_output_nucleolus_tracked_starved.csv'
nucleus_fed = 'AllPos_acdc_output_nucleus_fed.csv'
nucleus_str = 'AllPos_acdc_output_nucleus_starved.csv'

nucleolus_fed = os.path.join(data_path, dataset_name, nucleolus_fed)
nucleolus_str = os.path.join(data_path, dataset_name, nucleolus_str)
nucleus_fed = os.path.join(data_path, dataset_name, nucleus_fed)
nucleus_str = os.path.join(data_path, dataset_name, nucleus_str)

df_nucleolus_fed = pd.read_csv(nucleolus_fed, index_col=['Cell_ID', 'Position_n'])
df_nucleolus_str = pd.read_csv(nucleolus_str, index_col=['Cell_ID', 'Position_n'])
df_nucleus_fed = pd.read_csv(nucleus_fed, index_col=['Cell_ID', 'Position_n'])
df_nucleus_str = pd.read_csv(nucleus_str, index_col=['Cell_ID', 'Position_n'])

df_fed = df_nucleus_fed.join(
    df_nucleolus_fed, 
    rsuffix='_nucleolus', 
    lsuffix='_nucleus',
    how='inner'
)

print(df_fed[['cell_vol_vox_3D_nucleolus', 'cell_vol_vox_3D_nucleus']])

df_fed['cell_vol_nucleolus_percent_nucleus'] = (
    df_fed['cell_vol_vox_3D_nucleolus']
    / df_fed['cell_vol_vox_3D_nucleus']
)

print('Fed statistics:')
print(df_fed['cell_vol_nucleolus_percent_nucleus'].describe())

df_str = df_nucleus_str.join(
    df_nucleolus_str, 
    rsuffix='_nucleolus', 
    lsuffix='_nucleus',
    how='inner'
)

print(df_str[['cell_vol_vox_3D_nucleolus', 'cell_vol_vox_3D_nucleus']])

df_str['cell_vol_nucleolus_percent_nucleus'] = (
    df_str['cell_vol_vox_3D_nucleolus']
    / df_str['cell_vol_vox_3D_nucleus']
)

print('Starved statistics:')
print(df_str['cell_vol_nucleolus_percent_nucleus'].describe())
