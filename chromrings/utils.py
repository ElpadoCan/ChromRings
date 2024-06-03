import os
from natsort import natsorted

import pandas as pd

from chromrings import (
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    tables_path, USE_ABSOLUTE_DIST, USE_MANUAL_NUCLEOID_CENTERS, PLANE, 
    LARGEST_NUCLEI_PERCENT, MIN_LENGTH_PROFILE_PXL
)

def listdir(path):
    return natsorted([
        f for f in os.listdir(path)
        if not f.startswith('.')
        and not f == 'desktop.ini'
    ])

def get_pos_foldernames(exp_path):
    ls = listdir(exp_path)
    pos_foldernames = [
        pos for pos in ls if pos.find('Position_')!=-1
        and os.path.isdir(os.path.join(exp_path, pos))
        and os.path.exists(os.path.join(exp_path, pos, 'Images'))
    ]
    return pos_foldernames

def _read_df_profiles_from_prefix(filename_prefix, stat_to_plot='mean'):
    filename_prefix = filename_prefix.replace(
        'norm_mean_profile', f'norm_{stat_to_plot}_profile'
    )
    profiles_filename = f'{filename_prefix}_profiles.parquet'
    df_profiles_path = os.path.join(tables_path, profiles_filename)
    df_profiles = (
        pd.read_parquet(df_profiles_path).reset_index()
        .sort_values('dist_perc')
    )
    return df_profiles, profiles_filename

def read_df_profiles(
        batch_name=None, stat_to_plot='mean', load_absolute_dist=None
    ):
    if load_absolute_dist is None:
        load_absolute_dist = USE_ABSOLUTE_DIST
    
    if batch_name is None:
        from chromrings import batch_name
        
    try:
        filename_prefix = (
            f'{batch_name}'
            f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
            f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
            f'_norm_how_{NORMALISE_HOW}'
            f'_absolut_dist_{load_absolute_dist}'
            f'_manual_nucleolus_centers_{USE_MANUAL_NUCLEOID_CENTERS}'
            f'_{PLANE}plane'
        )
        if LARGEST_NUCLEI_PERCENT is not None:
            filename_prefix = filename_prefix.replace(
                f'_{PLANE}plane', 
                f'_only_largest_nuclei_perc_{int(LARGEST_NUCLEI_PERCENT*100)}'
                f'_{PLANE}plane'
            )
        if MIN_LENGTH_PROFILE_PXL > 0:
            filename_prefix = filename_prefix.replace(
                f'_{PLANE}plane', 
                f'_min_length_profile_pixels_{MIN_LENGTH_PROFILE_PXL}'
                f'_{PLANE}plane'
            )
        
        df_profiles, profiles_filename = _read_df_profiles_from_prefix(
            filename_prefix, stat_to_plot=stat_to_plot
        )
        return df_profiles, profiles_filename 
    except Exception as e:
        pass
    
    try:
        filename_prefix = (
            f'{batch_name}'
            f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
            f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
            f'_norm_how_{NORMALISE_HOW}'
            f'_absolut_dist_{load_absolute_dist}'
            f'_manual_nucleolus_centers_{USE_MANUAL_NUCLEOID_CENTERS}'
            f'_{PLANE}plane'
        )
        df_profiles, profiles_filename = _read_df_profiles_from_prefix(
            filename_prefix, stat_to_plot=stat_to_plot
        )
        return df_profiles, profiles_filename 
    except Exception as e:
        pass
    
    try:
        filename_prefix = (
            f'{batch_name}'
            f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
            f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
            f'_norm_how_{NORMALISE_HOW}'
            f'_absolut_dist_{USE_ABSOLUTE_DIST}'
            f'_manual_nucleolus_centers_{USE_MANUAL_NUCLEOID_CENTERS}'
        )
        df_profiles, profiles_filename = _read_df_profiles_from_prefix(
            filename_prefix, stat_to_plot=stat_to_plot
        )
        return df_profiles, profiles_filename 
    except Exception as e:
        pass
    
    try:
        filename_prefix = (
            f'{batch_name}'
            f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
            f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
            f'_norm_how_{NORMALISE_HOW}'
            f'_absolut_dist_{USE_ABSOLUTE_DIST}'
        )
        df_profiles, profiles_filename = _read_df_profiles_from_prefix(
            filename_prefix, stat_to_plot=stat_to_plot
        )
        return df_profiles, profiles_filename 
    except Exception as e:
        pass
    
    try:
        filename_prefix = (
            f'{batch_name}'
            f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
            f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
            f'_norm_how_{NORMALISE_HOW}'
        )
        df_profiles, profiles_filename = _read_df_profiles_from_prefix(
            filename_prefix, stat_to_plot=stat_to_plot
        )
        return df_profiles, profiles_filename 
    except Exception as e:
        raise e