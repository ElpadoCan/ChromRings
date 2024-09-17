import os
import json
import re

from tqdm import tqdm

import numpy as np
import skimage.io
import skimage.measure

from chromrings import data_path, figures_path
from chromrings import (
    data_info_json_path, utils, USE_ABSOLUTE_DIST
)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_pdf import PdfPages

SAVE = True

STAT_TO_PLOT = 'mean'
EXP_TO_PLOT = None # 'Pol III-auxin 3hrs' # 'Pol I-refed'

def plot(batch_name):
    df_profiles, profiles_filename = utils.read_df_profiles(
        batch_name=batch_name, stat_to_plot=STAT_TO_PLOT
    )
    df_profiles = df_profiles.set_index('dist_perc')
    print(f'Using table file "{profiles_filename}"')

    with open(data_info_json_path, 'r') as json_file:
        data_info = json.load(json_file)
        
    batch_info = data_info[batch_name]
    exp_foldernames = batch_info['experiments']
    channel_names = batch_info['channel']
    batch_path = os.path.join(data_path, *batch_info['folder_path'].split('/')[1:])
    pdf_filepath = os.path.join(
        figures_path, f'{batch_name}_{EXP_TO_PLOT}_profiles_with_image.pdf'
    )
    pdf = PdfPages(pdf_filepath)
    exp_info = {}
    for e, exp in enumerate(exp_foldernames):
        if EXP_TO_PLOT is not None and exp != EXP_TO_PLOT:
            continue
        exp_path = os.path.join(batch_path, *exp.split('/'))
        if not isinstance(channel_names, str):
            channel_name = channel_names[e]
        else:
            channel_name = channel_names
        exp_info[exp] = {}
        df_profiles_exp = df_profiles[exp]
        pos_foldernames = (
            df_profiles_exp.columns.get_level_values(0).unique().to_list()
        )
        for pos in tqdm(pos_foldernames, ncols=100):
            images_path = os.path.join(exp_path, pos, 'Images')
            for file in utils.listdir(images_path):
                file_path = os.path.join(images_path, file)
                if file.endswith(f'{channel_name}.tif'):
                    img = skimage.io.imread(file_path)
                elif file.endswith('segm.npz'):
                    lab = np.load(file_path)['arr_0']
            
            img_data_max = img.max()
            intensity_image = -img + img_data_max
            rp = skimage.measure.regionprops(lab, intensity_image=intensity_image)
            rp = {obj.label:obj for obj in rp}
            
            fig, ax = plt.subplots(3, 4, figsize=(16, 10))
            ax = ax.flatten()
            df_profiles_pos = df_profiles_exp[pos]
            lab_3D = lab.copy()
            for c, col in enumerate(df_profiles_pos.columns):
                ID = int(re.findall(r'ID_(\d+)_mean_radial_profile', col)[0])
                obj = rp[ID]
                yc_local, xc_local = obj.centroid_weighted_local[-2:]
                if len(obj.weighted_centroid) == 3:
                    zc, yc, xc = obj.weighted_centroid
                    lab_3D[:] = 0
                    lab_3D[obj.slice][obj.image] = ID
                    lab_2D = lab_3D[round(zc)]
                    img_2D = img[round(zc)]
                else:
                    lab_2D = lab_3D
                    img_2D = img
                obj_2D = skimage.measure.regionprops(lab_2D)[0]
                obj_intens_img = img_2D[obj_2D.slice]
                if c > 11:
                    break
                axis = ax[c]
                inset_ax = inset_axes(
                    axis, width='35%', height='35%', loc='lower center'
                )
                inset_ax.axis('off')
                inset_ax.imshow(obj_intens_img)
                inset_ax.plot(xc_local, yc_local, 'r.')
                df_profiles_ID = df_profiles_pos[col]
                axis.plot(df_profiles_ID.index, df_profiles_ID.values)
                axis.set_title(f'ID = {ID}')
            fig.suptitle(f'{exp} - {pos}')
            pdf.savefig(fig)
            plt.close(fig=fig)

    pdf.close()
    # mng = plt.get_current_fig_manager()
    # mng.window.showMaximized()
    plt.close()
    print('='*100)
    print(f'Done. Plots with images saved at "{pdf_filepath}"')
    print('='*100)
    try:
        utils.open_file(pdf_filepath)
    except Exception as err:
        pass

if __name__ == '__main__':
    batch_names = (
        '24h recovery', 
    )
    for batch_name in batch_names:
        plot(batch_name)