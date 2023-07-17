import os
import json

import numpy as np
import pandas as pd

import skimage.io
import skimage.measure
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from tqdm import tqdm

from chromrings import (
    data_path, core, utils, tables_path, data_info_json_path,
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    batch_name, USE_ABSOLUTE_DIST, USE_MANUAL_NUCLEOID_CENTERS,
    PLANE
)

np.seterr(all='raise')

SAVE = True
INSPECT_SINGLE_PROFILES = False
INSPECT_MEAN_PROFILE = False

if not USE_ABSOLUTE_DIST:
    resample_bin_size_dist = 5
else:
    resample_bin_size_dist = 1

with open(data_info_json_path, 'r') as json_file:
    data_info = json.load(json_file)

batch_info = data_info[batch_name]

channel_name = batch_info['channel']
batch_folder = batch_info['folder_path']
exp_foldernames = batch_info['experiments']

print('*'*60)
answer = input(
    'You are about to analyse the following experiments:\n'
    f'  * Name: {batch_name}\n'
    f'  * Folders: {exp_foldernames}\n'
    f'  * Channel: {channel_name}\n'
    f'  * Plane: {PLANE}\n'
    '\n'
    'Continue ([Y]/N)? '
)
if answer.lower() == 'n':
    exit('Execution stopped')

if not SAVE:
    while True:
        answer = input(
            'Are you sure you do not want to SAVE? '
            '1) Yes, do NOT save. 2) No, DO save. q) Quit. : '
        )
        if answer.lower() == 'q':
            exit('Execution stopped by the user.')
        elif answer == '1':
            SAVE = False
            break
        elif answer == '2':
            SAVE = True
            break
        else:
            print(
                f'{answer} is not a valid answer. '
                'Enter 1 or 2 to chosse an option or q to quit.'
            )

batch_dfs = []
batch_dfs_skew = []
batch_dfs_CV = []
batch_dfs_std = []

all_profiles_summary_df = []
keys = []

main_pbar = tqdm(total=len(exp_foldernames), ncols=100)
for e, exp_folder in enumerate(exp_foldernames):
    main_pbar.set_description(f'Analysing {exp_folder}')
    exp_rel_path_parts = exp_folder.split('/')
    exp_path = os.path.join(os.path.abspath(batch_folder), *exp_rel_path_parts)
    pos_foldernames = utils.get_pos_foldernames(exp_path)
    if isinstance(channel_name, str):
        channel = channel_name
    else:
        channel = channel_name[e]
    for position_n in tqdm(pos_foldernames, ncols=100, desc='Position', leave=False):
        images_path = os.path.join(exp_path, position_n, 'Images')
        segm_filename = None
        image_filename = None
        nucleolus_segm_filename = None
        nucleolus_centers_csv_filename = None
        for file in utils.listdir(images_path):
            if file.endswith(f'{channel}.tif'):
                image_filename = file
            elif file.endswith('segm_nucleolus.npz'):
                nucleolus_segm_filename = file
            # elif file.find('_segm_') != -1 and file.endswith('.npz'):
            #     segm_filename = file
            elif file.endswith('segm.npz'):
                segm_filename = file
            elif file.endswith('nu.csv'):
                nucleolus_centers_csv_filename = file

        if image_filename is None:
            main_pbar.close()
            raise FileNotFoundError(
                'The following experiment does not have the file ending with '
                f'"{channel}.tif": "{images_path}"'
            )
        if segm_filename is None:
            main_pbar.close()
            raise FileNotFoundError(
                'The following experiment does not have a the file ending with '
                f'"segm.npz": "{images_path}"'
            )
        if nucleolus_segm_filename is None:
            nucleolus_segm_data = None
        else:
            nucleolus_segm_filepath = os.path.join(
                images_path, nucleolus_segm_filename
            )
            nucleolus_segm_data = np.load(nucleolus_segm_filepath)['frame_0']
        
        nucleolus_centers_df = None
        if nucleolus_centers_csv_filename is not None and USE_MANUAL_NUCLEOID_CENTERS:
            nucleolus_centers_csv_filepath = os.path.join(
                images_path, nucleolus_centers_csv_filename
            )
            nucleolus_centers_df = (
                pd.read_csv(nucleolus_centers_csv_filepath)
                .set_index('Cell_ID')
            )

        segm_filepath = os.path.join(images_path, segm_filename)
        image_filepath = os.path.join(images_path, image_filename)

        segm_data = np.load(segm_filepath)['arr_0']
        img_data = skimage.io.imread(image_filepath)

        rp = core.radial_profiles(
            segm_data, img_data, 
            how='object', 
            plane=PLANE,
            invert_intensities=True, 
            resample_bin_size_dist=resample_bin_size_dist,
            extra_radius=0,
            tqdm_kwargs={'position': 2, 'leave': False, 'ncols': 100},
            normalize_every_profile=NORMALIZE_EVERY_PROFILE,
            normalise_average_profile=NORMALISE_AVERAGE_PROFILE,
            normalise_how=NORMALISE_HOW,
            inspect_single_profiles=INSPECT_SINGLE_PROFILES,
            inspect_mean_profile=INSPECT_MEAN_PROFILE,
            inner_lab=nucleolus_segm_data,
            use_absolute_dist=USE_ABSOLUTE_DIST,
            centers_df=nucleolus_centers_df
        )

        IDs = []
        argmeans = []
        argmaxs = []
        obj_series = []
        obj_series_skew = []
        obj_series_CV = []
        obj_series_std = []
        stds = []
        for obj in rp:
            obj_series.append(obj.mean_radial_profile)    
            obj_series_skew.append(obj.skews_radial_profile) 
            obj_series_CV.append(obj.CVs_radial_profile)  
            obj_series_std.append(obj.stds_radial_profile)           
            IDs.append(obj.label)
            argmeans.append(obj.radial_profile_argmean)
            argmaxs.append(obj.radial_profile_argmax)
            stds.append(obj.radial_profile_distr_std)

        df_profile = pd.concat(obj_series, axis=1)
        df_profile.index.name = 'dist_perc'
        batch_dfs.append(df_profile)
        
        df_profile_CV = pd.concat(obj_series_CV, axis=1)
        df_profile_CV.index.name = 'dist_perc'
        batch_dfs_CV.append(df_profile_CV)
        
        df_profile_skew = pd.concat(obj_series_skew, axis=1)
        df_profile_skew.index.name = 'dist_perc'
        batch_dfs_skew.append(df_profile_skew)
        
        df_profile_std = pd.concat(obj_series_std, axis=1)
        df_profile_std.index.name = 'dist_perc'
        batch_dfs_std.append(df_profile_std)

        summary_df = pd.DataFrame({
            'ID': IDs, 
            'argmean': argmeans,
            'argmax': argmaxs,
            'std': stds
        }).set_index('ID')

        summary_df['CV'] = summary_df['std'] / summary_df['argmean']

        all_profiles_summary_df.append(summary_df)
        key = (exp_folder, position_n)
        keys.append(key)

        # if condition == 'str':
        #     fig, ax = plt.subplots(3, 4)
        #     ax = ax.flatten()
        #     for o, obj in enumerate(rp):
        #         if o >= len(ax):
        #             break
                
        #         xx == obj.mean_radial_profile.index
        #         yy = obj.mean_radial_profile.values
        #         ax[o].plot(xx, yy)
        #     plt.show()
    main_pbar.update()
main_pbar.close()

batch_df_profile = pd.concat(batch_dfs, axis=1, keys=keys)
batch_df_profile.index.name = 'dist_perc'

batch_df_profile_skew = pd.concat(batch_dfs_skew, axis=1, keys=keys)
batch_df_profile_skew.index.name = 'dist_perc'

batch_df_profile_CV = pd.concat(batch_dfs_CV, axis=1, keys=keys)
batch_df_profile_CV.index.name = 'dist_perc'

batch_df_profile_std = pd.concat(batch_dfs_std, axis=1, keys=keys)
batch_df_profile_std.index.name = 'dist_perc'

final_df_summary = pd.concat(
    all_profiles_summary_df, keys=keys, 
    names=[*[f'folder_level_{i}' for i in range(len(keys[0]))], 'Cell_ID']
)

exp_foldername = '_'.join(exp_folder)
filename_prefix = (
    f'{batch_name}'
    f'_norm_single_profile_{NORMALIZE_EVERY_PROFILE}'
    f'_norm_mean_profile_{NORMALISE_AVERAGE_PROFILE}'
    f'_norm_how_{NORMALISE_HOW}'
    f'_absolut_dist_{USE_ABSOLUTE_DIST}'
    f'_manual_nucleolus_centers_{USE_MANUAL_NUCLEOID_CENTERS}'
    f'_{PLANE}plane'
)

filename_prefix_skew = filename_prefix.replace(
    'norm_mean_profile', 'norm_skew_profile'
)
filename_prefix_CV = filename_prefix.replace(
    'norm_mean_profile', 'norm_CV_profile'
)
filename_prefix_std = filename_prefix.replace(
    'norm_mean_profile', 'norm_std_profile'
)

if SAVE:
    df_profile_path = os.path.join(tables_path, f'{filename_prefix}_profiles.parquet')
    batch_df_profile.to_parquet(df_profile_path)
    
    df_profile_path_skew = os.path.join(tables_path, f'{filename_prefix_skew}_profiles.parquet')
    batch_df_profile_skew.to_parquet(df_profile_path_skew)
    
    df_profile_path_CV = os.path.join(tables_path, f'{filename_prefix_CV}_profiles.parquet')
    batch_df_profile_CV.to_parquet(df_profile_path_CV)
    
    df_profile_path_std = os.path.join(tables_path, f'{filename_prefix_std}_profiles.parquet')
    batch_df_profile_std.to_parquet(df_profile_path_std)

    print(f'Profiles saved to "{df_profile_path}"')

    final_df_summary_path = os.path.join(tables_path, f'{filename_prefix}_profiles_summary.parquet')
    final_df_summary.to_parquet(final_df_summary_path)

    print(f'Summary df saved to "{final_df_summary_path}"')

