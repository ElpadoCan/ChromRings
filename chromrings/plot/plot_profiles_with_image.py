import os
import json
import re

from tqdm import tqdm

import pandas as pd
import numpy as np
import skimage.io
import skimage.measure

from chromrings import data_path, figures_path, data_info_json_path, utils
from chromrings.current_analysis import (
    USE_ABSOLUTE_DIST, USE_MANUAL_NUCLEOID_CENTERS
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
        figures_path, 
        f'{batch_name}_{EXP_TO_PLOT}_profiles_with_image'
        f'_manual_nucleolus_centers_{USE_MANUAL_NUCLEOID_CENTERS}'
        '.pdf'
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
            nucleolus_centers_csv_filename = None
            lab = None
            img = None
            for file in utils.listdir(images_path):
                file_path = os.path.join(images_path, file)
                if file.endswith(f'{channel_name}.tif'):
                    img = skimage.io.imread(file_path)
                elif file.endswith('segm.npz'):
                    lab = np.load(file_path)['arr_0']
                elif file.endswith('nu.csv'):
                    nucleolus_centers_csv_filename = file
            
            load_manual_centr = (
                nucleolus_centers_csv_filename is not None
                and USE_MANUAL_NUCLEOID_CENTERS
            )
            nucleolus_centers_df = None
            if load_manual_centr:
                nucleolus_centers_csv_filepath = os.path.join(
                    images_path, nucleolus_centers_csv_filename
                )
                nucleolus_centers_df = (
                    pd.read_csv(nucleolus_centers_csv_filepath)
                    .set_index('Cell_ID')
                    .dropna(axis=1)
                )
            
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
                zc = None
                if nucleolus_centers_df is not None:
                    # Center coordinates are provided as input
                    center_df = nucleolus_centers_df.loc[obj.label]                
                    yc = center_df['y']
                    xc = center_df['x']
                    weighted_centroid = (yc, xc)
                    try:
                        zc = center_df['z']
                        weighted_centroid = (zc, yc, xc)
                    except KeyError as err:
                        zc = None                    
                elif len(obj.weighted_centroid) == 3:
                    zc, yc, xc = obj.weighted_centroid
                else:
                    yc, xc = obj.weighted_centroid
                
                if zc is None:
                    lab_2D = lab_3D
                    img_2D = img
                    ymin, xmin, _, _ = obj.bbox
                else:
                    lab_3D[:] = 0
                    lab_3D[obj.slice][obj.image] = ID
                    lab_2D = lab_3D[round(zc)]
                    img_2D = img[round(zc)]
                    _, ymin, xmin, _, _, _ = obj.bbox
                
                xc_local, yc_local = xc-xmin, yc-ymin
                
                if c > 11:
                    break
                
                rp_2D = skimage.measure.regionprops(lab_2D)
                rp_2D_mapper = {obj.label:obj for obj in rp_2D}
                obj_2D = rp_2D_mapper[ID]
                obj_intens_img = img_2D[obj_2D.slice]
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
    from chromrings.current_analysis import batch_name
    plot(batch_name)
