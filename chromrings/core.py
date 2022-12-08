import numpy as np
import cv2
from math import sqrt, pow
import skimage.measure
from scipy.optimize import least_squares
from scipy.special import erf

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

def get_objContours(obj, obj_image=None, all=False):
    if all:
        retrieveMode = cv2.RETR_CCOMP
    else:
        retrieveMode = cv2.RETR_EXTERNAL
    if obj_image is None:
        obj_image = obj.image.astype(np.uint8)
    contours, _ = cv2.findContours(
        obj_image, retrieveMode, cv2.CHAIN_APPROX_NONE
    )
    if len(obj.bbox) > 4:
        # 3D object
        _, min_y, min_x, _, _, _ = obj.bbox
    else:
        min_y, min_x, _, _ = obj.bbox
    if all:
        return [np.squeeze(cont, axis=1)+[min_x, min_y] for cont in contours]
    cont = np.squeeze(contours[0], axis=1)
    cont = np.vstack((cont, cont[0]))
    cont += [min_x, min_y]
    return cont

def radial_profiles(
        lab, img_data, extra_radius=0, how='object', 
        invert_intensities=True, resample_bin_size_perc=5,
        tqdm_kwargs=None, normalize_profile=True
    ):
    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    sp = extra_radius
    if invert_intensities:
        img_data_max = img_data.max()
        intensity_image = -img_data + img_data_max
    else:
        intensity_image = img_data
    rp = skimage.measure.regionprops(lab, intensity_image=intensity_image)

    for obj in tqdm(rp, desc='Computing radiant profiles', **tqdm_kwargs):
        weighted_centroid = obj.weighted_centroid
        if len(weighted_centroid) == 3:
            zc, yc, xc = weighted_centroid
            lab_2D = lab.copy()
            lab_2D[lab_2D != obj.label] = 0
            lab_2D = lab_2D[round(zc)]
            obj_z = skimage.measure.regionprops(lab_2D)[0]
            img_data_2D = img_data[round(zc)]
        else:
            yc, xc = weighted_centroid
            obj_z = obj
            lab_2D = lab
            img_data_2D = img_data
        
        x1, y1 = round(xc), round(yc)
        ymin, xmin, ymax, xmax = obj_z.bbox
        h, w = ymax-ymin, xmax-xmin 
        radius = round(max(h, w)/2 + sp)

        if how == 'cirlce':
            rr, cc = skimage.draw.circle_perimeter(
                y1, x1, radius, shape=lab_2D.shape
            )
        else:
            dilated = lab_2D.copy()
            footprint = skimage.morphology.disk(extra_radius)
            dilated = skimage.morphology.dilation(dilated, footprint)
            dilated_lab = skimage.measure.label(dilated)
            dilated_ID = dilated_lab[y1, x1]
            dilated_lab[dilated_lab!=dilated_ID] = 0
            dilated_obj = skimage.measure.regionprops(dilated_lab)[0]
            dilated_contour = get_objContours(dilated_obj)
            rr, cc = dilated_contour[:,1], dilated_contour[:,0]
            # fig, ax = plt.subplots()
            # ax.imshow(lab_2D)
            # ax.plot(cc, rr)
            # plt.show()
            # import pdb; pdb.set_trace()

        obj.radial_profiles = []
        obj.resampled_radial_profiles = []
        all_dist = set()
        for r, (y2, x2) in enumerate(zip(rr, cc)):
            yy_line, xx_line = skimage.draw.line(y1, x1, y2, x2)
            vals = img_data_2D[yy_line, xx_line]

            # Normalize distance to object edge
            full_dist = sqrt(pow(y2-y1, 2) + pow(x2-x1, 2))
            dist_100 = full_dist - extra_radius
            diff = np.subtract(np.column_stack((xx_line, yy_line)), (x1, y1))
            dist = np.linalg.norm(diff, axis=1)/dist_100
            dist_perc = np.round(dist*100).astype(int)
            obj.radial_profiles.append(
                {'x': xx_line, 'y': yy_line, 'values': vals, 'norm_dist': dist_perc}
            )

            if normalize_profile:
                vals = vals/vals.max()
            
            if resample_bin_size_perc > 0:
                _df = pd.DataFrame(
                        {'dist_perc': dist_perc, 'value': vals}
                    ).set_index('dist_perc')
                
                _df.index = pd.to_datetime(_df.index)
                rs = f'{resample_bin_size_perc}ns'
                resampled = _df.resample(rs, label='right').mean().dropna()
                dist_perc = resampled.index.astype(np.int64)
                obj.resampled_radial_profiles.append(
                    {'values': resampled['value'].values, 'norm_dist': dist_perc}
                )
                vals = resampled['value'].values

            all_dist.update(dist_perc)

            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(img_data_2D)
            # ax[0].set_xlim((xmin-sp, xmax+sp))
            # ax[0].set_ylim((ymin-sp, ymax+sp))
            # ax[0].plot(xx_line, yy_line, 'r.')
            # ax[1].plot(dist_perc, vals)
            # plt.show()

        if not resample_bin_size_perc > 0:
            obj.resampled_radial_profiles = obj.radial_profiles


        cols = [f'value_{r}' for r in range(len(obj.resampled_radial_profiles))]
        obj.radial_df = pd.DataFrame(
            index=list(all_dist), columns=cols
        )
        for r, profile in enumerate(obj.resampled_radial_profiles):
            idx = profile['norm_dist']
            # obj.radial_df[f'value_{r}'] = np.nan
            obj.radial_df.loc[idx, f'value_{r}'] = profile['values']
        
        obj.radial_df.sort_index(inplace=True)

        obj.mean_radial_profile = obj.radial_df.mean(axis=1)
        obj.mean_radial_profile.name = f'ID_{obj.label}_mean_radial_profile'

        # fig, ax = plt.subplots(1,2)
        # ax[0].imshow(img_data_2D)
        # ax[0].set_xlim((xmin-sp, xmax+sp))
        # ax[0].set_ylim((ymin-sp, ymax+sp))
        # ax[1].plot(obj.mean_radial_profile.index, obj.mean_radial_profile.values)
        # plt.show()

        '''Describe mean_radial_profile distribution'''
        x = obj.mean_radial_profile.index.values/100
        obj.radial_profile_argmax = obj.mean_radial_profile.idxmax()/100


        tot_samples = obj.mean_radial_profile.values.sum()
        frequency = obj.mean_radial_profile.values
        density = frequency/tot_samples
        obj.radial_profile_argmean = (density*x).sum()
        
        residuals = frequency*(np.square(x-obj.radial_profile_argmean))
        variance = np.sum(residuals)/(tot_samples-1)
        std = sqrt(variance)
        
        obj.radial_profile_distr_std = std
            
    return rp

class PeaksModel:
    def __init__(self, n_peaks=2):
        self.n_peaks = n_peaks

    def gauss_function(self, X, x0, sx, A, B=0):
        # Center rotation around peak center
        xc = X - x0
        # Build 3D gaussian by multiplying each 1D gaussian function
        gauss_x = np.exp(-(xc**2)/(2*(sx**2)))
        f = A*gauss_x
        return f
    
    def get_init_guess(self, xx_peaks, yy_peaks, X):
        self.n_peaks = len(xx_peaks)
        init_guess = []
        sx = np.std(X)/len(xx_peaks)
        for x_peak, y_peak in zip(xx_peaks, yy_peaks):
            A_peak = y_peak * sx * np.sqrt(2*np.pi)
            init_guess.extend((x_peak, sx, A_peak))
        return init_guess

    def model(self, X, coeffs, B):
        _model = np.zeros(len(X))
        n = 0
        for _ in range(self.n_peaks):
            x0, sx, A = coeffs[n:n+3]
            _model += self.gauss_function(X, x0, sx, A)
            n += 3
        return _model + B

    def fit(self, X, Y, init_guess_coeffs, B_guess):
        coeffs = init_guess_coeffs.copy()
        coeffs.append(B_guess)
        result = least_squares(
            self.residuals, coeffs, args=(Y, X)
        )
        return result
    
    def residuals(self, coeffs, Y, X):
        B = coeffs[-1]
        coeffs = coeffs[:-1]
        _model = self.model(X, coeffs, B)
        return Y - _model
    
    def integrate(self, coeffs, B):
        # Use 95% as integration area
        n = 0
        Is_foregr = np.empty(self.n_peaks, dtype=np.float64)
        Is_tot = np.empty(self.n_peaks, dtype=np.float64)
        for i in range(self.n_peaks):
            x0, sx, A = coeffs[n:n+3]

            sx = abs(sx)
            
            xleft = -1.96 * sx
            xright = 1.96 * sx

            # Substitute variable x --> t to apply erf
            tleft = xleft / (np.sqrt(2)*sx)
            tright = xright / (np.sqrt(2)*sx)
            sx_t = sx * np.sqrt(np.pi/2)

            # Apply erf function
            D_erf = erf(tright)-erf(tleft)

            I_foregr = A * sx_t * D_erf

            Is_foregr[i] = I_foregr
            Is_tot[i] = I_foregr + B*(xright-xleft)

            n += 3
        return Is_foregr.sum(), Is_tot.sum(), Is_foregr, Is_tot