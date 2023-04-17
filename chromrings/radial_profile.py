import os
import json

import numpy as np
import pandas as pd

import skimage.io
import skimage.measure
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math import cos, sin, pi

from tqdm import tqdm

from chromrings import (
    data_path, core, utils, tables_path, data_info_json_path,
    NORMALIZE_EVERY_PROFILE, NORMALISE_AVERAGE_PROFILE, NORMALISE_HOW,
    batch_name
)

SAVE = True
INSPECT = False

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
    '\n'
    'Continue ([Y]/N)? '
)
if answer.lower() == 'n':
    exit('Execution stopped')

batch_dfs = []

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
        for file in utils.listdir(images_path):
            if file.endswith(f'{channel}.tif'):
                image_filename = file
            elif file.find('_segm_') != -1 and file.endswith('.npz'):
                segm_filename = file
            elif file.endswith('segm.npz'):
                segm_filename = file

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

        segm_filepath = os.path.join(images_path, segm_filename)
        image_filepath = os.path.join(images_path, image_filename)

        segm_data = np.load(segm_filepath)['arr_0']
        img_data = skimage.io.imread(image_filepath)

        rp = core.radial_profiles(
            segm_data, img_data, 
            how='object', 
            invert_intensities=True, 
            resample_bin_size_perc=5,
            extra_radius=0,
            tqdm_kwargs={'position': 2, 'leave': False, 'ncols': 100},
            normalize_every_profile=NORMALIZE_EVERY_PROFILE,
            normalise_average_profile=NORMALISE_AVERAGE_PROFILE,
            normalise_how=NORMALISE_HOW,
            inspect=INSPECT
        )

        IDs = []
        argmeans = []
        argmaxs = []
        obj_series = []
        stds = []
        for obj in rp:
            obj_series.append(obj.mean_radial_profile)                
            IDs.append(obj.label)
            argmeans.append(obj.radial_profile_argmean)
            argmaxs.append(obj.radial_profile_argmax)
            stds.append(obj.radial_profile_distr_std)

        df_profile = pd.concat(obj_series, axis=1)
        df_profile.index.name = 'dist_perc'
        batch_dfs.append(df_profile)

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
)

if SAVE:
    df_profile_path = os.path.join(tables_path, f'{filename_prefix}_profiles.parquet')
    batch_df_profile.to_parquet(df_profile_path)

    print(f'Profiles saved to "{df_profile_path}"')

    final_df_summary_path = os.path.join(tables_path, f'{filename_prefix}_profiles_summary.parquet')
    final_df_summary.to_parquet(final_df_summary_path)

    print(f'Summary df saved to "{final_df_summary_path}"')

