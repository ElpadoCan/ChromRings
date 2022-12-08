import os

import numpy as np
import pandas as pd

import skimage.io
import skimage.measure
import skimage.draw

import matplotlib.pyplot as plt
import matplotlib.patches as patches

from math import cos, sin, pi

from tqdm import tqdm

from chromrings import data_path, core, utils, tables_path

exp_folders = ['0_test_3D_Nada_24-11-2022']
channel_name = 'w1SDC488'

all_df_profile_fed = []
all_df_profile_starved = []

all_profiles_summary_df = []
keys = []

for exp_folder in exp_folders:
    for condition in ['str', 'fed']:
        print(f'Analysing {exp_folder}, condition: "{condition}"...')
        exp_path = os.path.join(data_path, exp_folder, condition)
        pos_foldernames = utils.get_pos_foldernames(exp_path)
        for position_n in tqdm(pos_foldernames, ncols=100, desc='Position'):
            images_path = os.path.join(exp_path, position_n, 'Images')
            for file in utils.listdir(images_path):
                if file.endswith(f'{channel_name}.tif'):
                    image_filename = file
                elif file.find('_segm_') != -1 and file.endswith('.npz'):
                    segm_filename = file

            segm_filepath = os.path.join(images_path, segm_filename)
            image_filepath = os.path.join(images_path, image_filename)

            segm_data = np.load(segm_filepath)['arr_0']
            img_data = skimage.io.imread(image_filepath)

            rp = core.radial_profiles(
                segm_data, img_data, 
                how='dilation', 
                invert_intensities=True, 
                resample_bin_size_perc=5,
                extra_radius=0,
                tqdm_kwargs = {'position': 1, 'leave': False, 'ncols': 100},
                normalize_profile=True
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

            df_profile = pd.concat(obj_series, axis=1, names=['dist_perc'])
            if condition == 'str':
                all_df_profile_starved.append(df_profile)
            else:
                all_df_profile_fed.append(df_profile)

            summary_df = pd.DataFrame({
                'ID': IDs, 
                'argmean': argmeans,
                'argmax': argmaxs,
                'std': stds
            }).set_index('ID')

            summary_df['CV'] = summary_df['std'] / summary_df['argmean']

            all_profiles_summary_df.append(summary_df)
            keys.append((exp_folder, condition, position_n))

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

df_profile_starved = pd.concat(all_df_profile_starved, axis=1)
df_profile_starved.index.name = 'dist_perc'

df_profile_fed = pd.concat(all_df_profile_fed, axis=1)
df_profile_fed.index.name = 'dist_perc'

final_df_summary = pd.concat(
    all_profiles_summary_df, keys=keys, 
    names=['experiment', 'condition', 'Position_n', 'ID']
)

df_profile_starved_path = os.path.join(tables_path, 'profiles_starved.csv')
df_profile_starved.to_csv(df_profile_starved_path)

df_profile_fed_path = os.path.join(tables_path, 'profiles_fed.csv')
df_profile_fed.to_csv(df_profile_fed_path)

final_df_summary_path = os.path.join(tables_path, 'profiles_summary.csv')
final_df_summary.to_csv(final_df_summary_path)

