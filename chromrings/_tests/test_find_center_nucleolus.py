import os
import json
import re

from tqdm import tqdm

import numpy as np
import skimage.io
import skimage.measure
import skimage.feature
import skimage.filters
import skimage.transform
import skimage.draw
import skimage.color

from chromrings import data_path, figures_path
from chromrings import (
    data_info_json_path, batch_name, utils, USE_ABSOLUTE_DIST
)

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import matplotlib
matplotlib.use('qtagg')

from matplotlib.backends.backend_pdf import PdfPages

SAVE = False
METHOD = 'blob_detection' # 'hough_circle' # 'peak_local_max'
STAT_TO_PLOT = 'mean'
EXP_TO_PLOT = None # 'Pol III-auxin 3hrs' # 'Pol I-refed'

df_profiles, profiles_filename = utils.read_df_profiles(stat_to_plot=STAT_TO_PLOT)
df_profiles = df_profiles.set_index('dist_perc')
print(f'Using table file "{profiles_filename}"')

with open(data_info_json_path, 'r') as json_file:
    data_info = json.load(json_file)
    
batch_info = data_info[batch_name]
exp_foldernames = batch_info['experiments']
channel_names = batch_info['channel']
batch_path = os.path.join(data_path, batch_name)
pdf = PdfPages(os.path.join(
    figures_path, f'{batch_name}_{EXP_TO_PLOT}_nucleolus_center_images.pdf')
)
exp_info = {}
footprint = np.ones((5,5), dtype=bool)
for e, exp in enumerate(exp_foldernames):
    if EXP_TO_PLOT is not None and exp != EXP_TO_PLOT:
        continue
    exp_path = os.path.join(batch_path, exp)
    try:
        channel_name = channel_names[e]
    except Exception as e:
        channel_name = channel_names
    exp_info[exp] = {}
    df_profiles_exp = df_profiles[exp]
    pos_foldernames = df_profiles_exp.columns.get_level_values(0).unique().to_list()
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
            axis = ax[c]
            
            ID = int(re.findall(r'ID_(\d+)_mean_radial_profile', col)[0])
            obj = rp[ID]
            zc, yc, xc = obj.weighted_centroid
            _, yc_local, xc_local = obj.centroid_weighted_local
            lab_3D[:] = 0
            lab_3D[obj.slice][obj.image] = ID
            lab_2D = lab_3D[round(zc)]
            obj_2D = skimage.measure.regionprops(lab_2D)[0]
            obj_intens_img = img[round(zc)][obj_2D.slice]
            
            detect_img = intensity_image[round(zc)][obj_2D.slice]

            if METHOD == 'peak_local_max':
                detect_img = skimage.filters.gaussian(detect_img, 3)
                
                local_max_coords = skimage.feature.peak_local_max(
                    detect_img, labels=obj_2D.image.astype(np.uint8),
                    num_peaks=1, exclude_border=10, footprint=footprint
                )
                y_peak, x_peak = local_max_coords[0]
                
                # Draw peak location
                axis.imshow(obj_intens_img)
                # axis.plot(xc_local, yc_local, 'r.')
                axis.plot(x_peak, y_peak, 'r.')
            elif METHOD == 'hough_circle':
                # Nucleus average axis length
                nucleus_radius = int(
                    (obj_2D.axis_major_length+obj_2D.axis_minor_length)/2
                )/2
                nucleolus_radius = int(nucleus_radius/3)
                
                # edges = skimage.filters.sobel(detect_img, mask=obj_2D.image)
                
                # Detect two radii
                hough_radii = np.arange(nucleolus_radius, nucleus_radius, 2)
                hough_res = skimage.transform.hough_circle(
                    obj_intens_img, hough_radii
                )
                
                # Select the most prominent 3 circles
                accums, cx, cy, radii = skimage.transform.hough_circle_peaks(
                    hough_res, hough_radii, total_num_peaks=2
                )
                
                # Draw circles
                obj_intens_img = (obj_intens_img/obj_intens_img.max()*255).astype(np.uint8)
                image = skimage.color.gray2rgb(obj_intens_img)
                for center_y, center_x, radius in zip(cy, cx, radii):
                    circy, circx = skimage.draw.circle_perimeter(
                        center_y, center_x, int(radius), shape=image.shape
                    )
                    image[circy, circx] = (220, 20, 20)
                axis.imshow(image)
                # import pdb; pdb.set_trace()
            elif METHOD == 'blob_detection':   
                detect_img = detect_img/detect_img.max()        
                detect_img[~obj_2D.image] = detect_img.min()
                blob_log = skimage.feature.blob_log(
                    detect_img, min_sigma=4, max_sigma=30, num_sigma=10, 
                    threshold=0.05
                )
                axis.imshow(detect_img)
                import pdb; pdb.set_trace()
        import pdb; pdb.set_trace()
            
        fig.suptitle(f'{exp} - {pos}')
        pdf.savefig(fig)

pdf.close()
# mng = plt.get_current_fig_manager()
# mng.window.showMaximized()
plt.close()
            
            
        