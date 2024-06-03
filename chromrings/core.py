import os
import traceback

from collections import defaultdict

from typing import Literal

import numpy as np
import cv2
from math import sqrt, pow
import skimage.measure
from scipy.optimize import least_squares, curve_fit
from scipy.special import erf
from scipy.signal import find_peaks
import math

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

from cellacdc.plot import imshow

pwd_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

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
        lab: np.ndarray, img_data: np.ndarray, 
        extra_radius=0, 
        how='object', 
        plane=Literal['xy', 'yz', 'xz'],
        invert_intensities=True, 
        resample_bin_size_dist=5,
        tqdm_kwargs=None, 
        normalize_every_profile=True,
        normalise_how='max', 
        normalise_average_profile=False, 
        inspect_single_profiles=False,
        inspect_mean_profile=False,
        inner_lab=None, 
        use_absolute_dist=False, 
        centers_df=None, 
        largest_nuclei_percent=None, 
        min_length_profile_pixels=0, 
        zeroize_inner_lab_edge=False
    ):
    """Compute the radial profile of single-cells. The final profile is the 
    average of all the possible profiles starting from weighted centroid to 
    each point on the segmentation object's contour.

    Parameters
    ----------
    lab : np.ndarray of ints
        Segmentation labels arrat
    img_data : np.ndarray
        Intensity image
    extra_radius : int, optional
        Grow object by `extra_radius` before getting its contour, by default 0
    how : str, optional
        Controls whether the contour should be the object's contour 
        (`how = 'object'`) or a  circle (`how = 'circle'`), by default 'object'
    plane : str, optional
        Plane to use. Options are 'xy', 'yz', or 'xz'
    invert_intensities : bool, optional
        Invert black/white levels in `img_data`, by default True
    resample_bin_size_dist : int, optional
        Bin size percentage of the distance from centroid to edge, by default 5
    tqdm_kwargs : dict, optional
        Keyword arguments to pass to tqdm progress bar, by default None
    normalize_every_profile : bool, optional
        Normalise every single profile by its own max, by default True
    normalise_average_profile : bool, optional
        Normalise final profile by its own max, by default False
    normalise_how : str, optional
        Which function to use for normalisation of final average profile, 
        either 'max' or 'sum', by default 'max'
    inspect_single_profiles : bool, optional
        Visualise partial results, by default False
    inspect_mean_profile : bool, optional
        Visualise mean profile, by default False
    inner_lab : np.ndarray, optional
        Optional inner segmentation labels, by default None. 
        If provided, it will be used to determine that 0% of the profile, 
        i.e., the outer edge of the object.
    use_absolute_dist : bool, optional
        Do not use percentage distances, by default False
    centers_df : pd.DataFrame, optional
        Pandas DataFrame with `z, y, x` columns and `Cell_ID` as index, 
        by default None.
        If not None, the z, y, x coordinates are used as the center of the 
        object for the radial profiles.
    largest_nuclei_percent : float, optional
        If not None, take only the largest nuclei, i.e., the 0.2 largest.
    min_length_profile_pixels : int, optional
        Minimum length of single profile that will be used to get the mean 
        single-cell profile. Default is 0
    zeroize_inner_lab_edge: bool, optional
        If `True`, the distance along the profile will be zeroised to the 
        edge of the `inner_lab`. 
    
    Returns
    -------
    list of `skimage.measure.RegionProperties`
        List of region properties. The additional properties computed by this 
        function are `mean_radial_profile`, `radial_profile_argmean`, 
        `radial_profile_argmax`, and `radial_profile_distr_std`.
    """    
    max_dtype = np.iinfo(img_data.dtype).max
    
    if plane == 'yz':
        lab = np.rot90(lab, axes=(0,2))
        img_data = np.rot90(img_data, axes=(0,2))
        if inner_lab is not None:
            inner_lab = np.rot90(inner_lab, axes=(0,2))
    elif plane == 'xz':
        lab = np.rot90(lab)
        img_data = np.rot90(img_data)
        if inner_lab is not None:
            inner_lab = np.rot90(inner_lab)

    if tqdm_kwargs is None:
        tqdm_kwargs = {}

    sp = extra_radius
    if invert_intensities:
        img_data_max = img_data.max()
        intensity_image = -img_data + img_data_max
    else:
        intensity_image = img_data
    rp = skimage.measure.regionprops(lab, intensity_image=intensity_image)

    if inner_lab is not None:
        inner_lab_masked = np.zeros_like(lab)

    min_area = None
    if largest_nuclei_percent:
        areas = [obj.area for obj in rp]
        min_area = np.quantile(areas, q=1-largest_nuclei_percent)
    
    for obj in tqdm(rp, desc='Computing radiant profiles', **tqdm_kwargs):
        if min_area is not None and obj.area < min_area:
            continue
        
        if inner_lab is not None:
             # Use centroid of the inner object provided as input
            inner_masked = inner_lab[obj.slice][obj.image]
            inner_masked = inner_masked[inner_masked>0]
            inner_ids, inner_count = np.unique(
                inner_masked, return_counts=True
            )
            max_count_idx = inner_count.argmax()    
            inner_id = inner_ids[max_count_idx]
            inner_lab_masked[:] = 0
            inner_lab_masked[inner_lab == inner_id] = obj.label
        
        if centers_df is not None and obj.label in centers_df.index:
            # Center coordinates are provided as input
            center_df = centers_df.loc[obj.label]
            zc = center_df['z']
            yc = center_df['y']
            xc = center_df['x']
            weighted_centroid = (zc, yc, xc)
        elif inner_lab is None:
            # Use weighted centroid
            weighted_centroid = obj.weighted_centroid
        else:
            inner_obj = skimage.measure.regionprops(inner_lab_masked)[0]
            _, yc, xc = inner_obj.centroid   
            zc, _, _ = obj.weighted_centroid
            weighted_centroid = (zc, yc, xc)
            
        if len(weighted_centroid) == 3:
            zc, yc, xc = weighted_centroid
            lab_2D = lab.copy()
            lab_2D[lab_2D != obj.label] = 0
            lab_2D = lab_2D[round(zc)]
            if inner_lab is not None:
                # inner_lab_2D = inner_lab_masked[round(zc)]
                inner_lab_2D = inner_lab_masked.max(axis=0)
            
            try:
                obj_z = skimage.measure.regionprops(lab_2D)[0]
            except Exception as err:
                import pdb; pdb.set_trace()
                continue
            img_data_2D = img_data[round(zc)]
        else:
            yc, xc = weighted_centroid
            obj_z = obj
            lab_2D = lab
            img_data_2D = img_data
            if inner_lab is not None:
                inner_lab_2D = inner_lab_masked
        
        # from cellacdc.plot import imshow
        # imshow(inner_lab_2D, lab_2D)
        # import pdb; pdb.set_trace()
        
        x1, y1 = round(xc), round(yc)
        ymin, xmin, ymax, xmax = obj_z.bbox
        h, w = ymax-ymin, xmax-xmin 
        radius = round(max(h, w)/2 + sp)

        if how == 'circle':
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
        
        if inner_lab is not None:
            inner_obj_2D = skimage.measure.regionprops(inner_lab_2D)[0]
            inner_contour = get_objContours(inner_obj_2D)
            inner_contours_lab = np.zeros(inner_lab_2D.shape, dtype=np.uint32)
            rr1 = np.arange(1, len(inner_contour)+1)
            rr2 = rr1 + rr1.max()
            inner_contours_lab[inner_contour[:,1], inner_contour[:,0]] = rr1
            inner_contours_lab[inner_contour[:,1], inner_contour[:,0]-1] = rr2

        obj.radial_profiles = []
        obj.resampled_radial_profiles = []
        all_dist = set()
        for r, (y2, x2) in enumerate(zip(rr, cc)):
            yy_line, xx_line = skimage.draw.line(y1, x1, y2, x2)
            vals = img_data_2D[yy_line, xx_line]
            if inner_lab is not None:
                inner_vals = inner_contours_lab[yy_line, xx_line]
                try:
                    inner_val = inner_vals[inner_vals>0][0]
                except Exception as e:
                    # fig, ax = plt.subplots()
                    # ax.imshow(inner_contours_lab)
                    # ax.plot(xx_line, yy_line)
                    # plt.show()
                    # import pdb; pdb.set_trace()
                    continue
                inner_yy, inner_xx = np.where(inner_contours_lab == inner_val)
                inner_y, inner_x = inner_yy[0], inner_xx[0]
                
            # Normalize distance to object edge
            if inner_lab is not None and zeroize_inner_lab_edge:
                y0, x0 = inner_y, inner_x
            else:
                y0, x0 = y1, x1
            xy_arr = np.column_stack((xx_line, yy_line))
            diff = np.subtract(xy_arr, (x0, y0))
            dist = np.linalg.norm(diff, axis=1)
            zero_dist_idx = dist.argmin()
            if not use_absolute_dist:
                full_dist = math.dist([y2, x2], [y0, x0]) 
                dist_100 = full_dist - extra_radius
                dist = dist/dist_100
                dist_perc = np.round(dist*100).astype(int)
            else:
                dist_perc = np.round(dist).astype(int)
                dist_perc_out_inner = dist_perc.copy()
            
            if min_length_profile_pixels > 0:
                profile_outer_len = math.dist([y2, x2], [y1, x1])
                if inner_lab is not None:
                    inner_len = math.dist([inner_y, inner_x], [y1, x1])
                    profile_outer_len -= inner_len
                if profile_outer_len < min_length_profile_pixels:
                    continue
            
            dist_perc[:zero_dist_idx] *= - 1
            obj.radial_profiles.append({
                'x': xx_line, 'y': yy_line, 'values': vals, 
                'norm_dist': dist_perc
            })

            if normalize_every_profile:
                vals = vals/vals.max()
            
            if resample_bin_size_dist > 0:
                _df = pd.DataFrame(
                        {'dist_perc': dist_perc, 'value': vals}
                    ).set_index('dist_perc')
                
                _df.index = pd.to_datetime(_df.index)
                rs = f'{resample_bin_size_dist}ns'
                resampled = _df.resample(rs, label='right').mean().dropna()
                dist_perc = resampled.index.astype(np.int64)
                obj.resampled_radial_profiles.append({
                    'values': resampled['value'].values, 
                    'norm_dist': dist_perc
                })
                vals = resampled['value'].values

            all_dist.update(dist_perc)

            if inspect_single_profiles:
                fig, ax = plt.subplots(1,2, figsize=(16,8))
                ax[0].imshow(img_data_2D)
                ax[0].set_xlim((xmin-sp, xmax+sp))
                ax[0].set_ylim((ymin-sp, ymax+sp))
                if inner_lab is not None:
                    ax[0].plot(inner_contour[:,0], inner_contour[:,1], 'r')
                ax[0].plot(xx_line, yy_line, 'r.')
                ax[1].plot(dist_perc, vals)
                figures_path = os.path.join(pwd_path, 'figures')
                fig_path = os.path.join(figures_path, f'{r:02d}_profile.png')
                fig.savefig(fig_path)
                plt.show()
                import pdb; pdb.set_trace()

        if not resample_bin_size_dist > 0:
            obj.resampled_radial_profiles = obj.radial_profiles


        cols = [f'value_{r}' for r in range(len(obj.resampled_radial_profiles))]
        obj.radial_df = pd.DataFrame(index=list(all_dist), columns=cols)
        obj.radial_df['is_saturated'] = False
        for r, profile in enumerate(obj.resampled_radial_profiles):
            idx = profile['norm_dist']
            # obj.radial_df[f'value_{r}'] = np.nan
            obj.radial_df.loc[idx, f'value_{r}'] = profile['values']
            is_saturated = np.max(profile['values']) == max_dtype
            obj.radial_df['is_saturated'] = is_saturated
        
        obj.radial_df.sort_index(inplace=True)
        obj.radial_df = obj.radial_df.astype(float)
        
        radial_df = obj.radial_df[cols]
        obj.mean_radial_profile = radial_df.mean(axis=1)
        if len(obj.mean_radial_profile) == 0:
            obj.mean_radial_profile = None
            continue
        
        obj.stds_radial_profile = radial_df.std(axis=1)
        obj.CVs_radial_profile = (
            (obj.stds_radial_profile/obj.mean_radial_profile)
            .replace(np.inf, np.nan)
        )
        obj.skews_radial_profile = radial_df.skew(axis=1)
        
        obj.mean_radial_profile.name = f'ID_{obj.label}_mean_radial_profile'
        obj.stds_radial_profile.name = f'ID_{obj.label}_CV_radial_profile'
        obj.CVs_radial_profile.name = f'ID_{obj.label}_std_radial_profile'
        obj.skews_radial_profile.name = f'ID_{obj.label}_skew_radial_profile'
        if normalise_average_profile:
            if normalise_how == 'tot_fluo':
                norm_value = img_data[obj.slice][obj.image].sum()
            elif normalise_how is None or normalise_how == 'None':
                try:
                    norm_value = np.iinfo(img_data.dtype).max
                except ValueError:
                    norm_value = 1
            else:
                norm_func = getattr(np, normalise_how)
                norm_value = norm_func(obj.mean_radial_profile)     
            obj.mean_radial_profile /= norm_value
            
        profile_series = (
            obj.stds_radial_profile, obj.CVs_radial_profile, 
            obj.skews_radial_profile
        )
        for series_profile in (profile_series):
            try:
                norm_func = getattr(np, normalise_how)
            except TypeError as e:
                break
            norm_value = norm_func(series_profile)
            series_profile /= norm_value
        
        if inspect_mean_profile:
            print('')
            print(obj.mean_radial_profile)
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(img_data_2D)
            ax[0].set_xlim((xmin-sp, xmax+sp))
            ax[0].set_ylim((ymin-sp, ymax+sp))
            if inner_lab is not None:
                ax[0].plot(inner_contour[:,0], inner_contour[:,1], 'r')
            ax[0].plot([x1], [y1], 'r.')
            ax[1].plot(obj.mean_radial_profile.index, obj.mean_radial_profile.values)
            plt.show()
            import pdb; pdb.set_trace()

        
        
        '''Describe mean_radial_profile distribution'''
        df_norm_distribution = obj.mean_radial_profile/obj.mean_radial_profile.max()
        x = df_norm_distribution.index.values/100
        try:
            obj.radial_profile_argmax = df_norm_distribution.idxmax()/100
        except Exception as err:
            import pdb; pdb.set_trace()

        tot_samples = df_norm_distribution.values.sum()
        frequency = df_norm_distribution.values
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
    
    def get_init_guess(self, xx_peaks, yy_peaks, X, A_init=1.0):
        self.n_peaks = len(xx_peaks)
        init_guess = []
        sx = np.std(X)/len(xx_peaks)
        for x_peak, y_peak in zip(xx_peaks, yy_peaks):
            # A_peak = y_peak * sx * np.sqrt(2*np.pi)
            init_guess.extend((x_peak, sx, A_init))
        return init_guess

    def bounds(self, yy, xx, init_guess, peak_center_max_range=5):
        lower_bounds = np.array([-np.inf]*(self.n_peaks*3 + 1))
        upper_bounds = np.array([np.inf]*(self.n_peaks*3 + 1))
        
        # Background and amplitude no lower than 0
        lower_bounds[-1] = 0
        lower_bounds[2::3] = 0
        
        # Force peak center at init guess +-5
        lower_bounds[:6:3] = np.array(init_guess)[::3] - peak_center_max_range
        upper_bounds[:6:3] = np.array(init_guess)[::3] + peak_center_max_range
        
        # Standard deviation no greater than xx range
        xx_range = np.max(xx) - np.min(xx)
        lower_bounds[1::3] = -xx_range
        upper_bounds[1::3] = xx_range

        # Background and amplitude no higher than max yy
        yy_max = np.max(yy)
        upper_bounds[-1] = yy_max
        upper_bounds[2::3] = yy_max
        
        return (lower_bounds, upper_bounds)
    
    def model(self, X, coeffs, B):
        _model = np.zeros(len(X))
        n = 0
        for _ in range(self.n_peaks):
            x0, sx, A = coeffs[n:n+3]
            _model += self.gauss_function(X, x0, sx, A)
            n += 3
        return _model + B

    def curve_fit_func(self, xdata, *coeffs):
        B = coeffs[-1]
        return self.model(xdata, coeffs, B)
    
    def least_squares(self, X, Y, init_guess_coeffs, B_guess, bounds=None):
        coeffs = init_guess_coeffs.copy()
        coeffs.append(B_guess)
        kwargs = {}
        if bounds is not None:
            kwargs['bounds'] = bounds
        try:
            result = least_squares(
                self.residuals, coeffs, args=(Y, X), **kwargs
            )
        except Exception as err:
            print('')
            traceback.print_exc()
            import pdb; pdb.set_trace()
            return 
        return result

    def fit(self, X, Y, init_guess_coeffs, B_guess, bounds=None):
        kwargs = {}
        if bounds is not None:
            kwargs['bounds'] = bounds
        coeffs = init_guess_coeffs.copy()
        coeffs.append(B_guess)
        coeffs, pcov = curve_fit(
            self.curve_fit_func, X, Y, coeffs, **kwargs
        )
        return coeffs, pcov
    
    def RMSE(self, yy, yy_pred):
        residuals = yy - yy_pred
        residuals_squared = np.square(residuals)
        num_samples = len(yy)
        residuals_squared_sum = np.sum(residuals_squared)
        RMSE = np.sqrt(residuals_squared_sum/num_samples)
        return RMSE
    
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

def fit_profiles(
        df_profiles, n_peaks=2, inspect=False, show_pbar=False, 
        init_guess_peaks_loc=None, A_init=1.0, peak_center_max_range=5
    ):
    n_profiles = len(df_profiles.columns)
    if show_pbar:
        pbar = tqdm(total=n_profiles, desc='Profile', ncols=100, leave=False)
    df_profiles_fit = df_profiles.copy()
    
    # Each peak has 3 coefficents plus a global background coeff --> 4 columns
    n_coeffs_per_profile = 3
    
    # We also save additional coeffs: B_pred and RMSE
    n_additional_coeffs = 2 
    coeffs_values = np.zeros(
        (n_profiles*n_peaks, n_coeffs_per_profile+n_additional_coeffs)
    )
    coeffs_idx = []
    model = PeaksModel(n_peaks=n_peaks)
    for i, column in enumerate(df_profiles.columns):
        df_profile = df_profiles[column]
        xx = df_profile.index.values
        yy = df_profile.values
        for j in range(n_peaks):
            coeffs_idx.append((*column, f'peak_{j}'))
        
        if init_guess_peaks_loc is None:
            peaks_idxs, props = find_peaks(yy, width=2)
            if len(peaks_idxs) != n_peaks:
                df_profiles_fit[column] = np.nan
                for j in range(n_peaks):
                    n = (n_coeffs_per_profile+n_additional_coeffs)
                    coeffs_values[(i*n_peaks)+j] = [np.nan]*n
                plt.plot(xx, yy)
                plt.scatter(xx[peaks_idxs], yy[peaks_idxs])
                plt.show()
                import pdb; pdb.set_trace()
                continue
        else:
            peaks_idxs = []
            for x_peak in init_guess_peaks_loc:
                abs_diff = np.abs(xx - x_peak)
                nearest_idx = np.where(abs_diff==sorted(abs_diff)[0])[0][0]
                peaks_idxs.append(nearest_idx)
        xx_peaks = xx[peaks_idxs]
        yy_peaks = yy[peaks_idxs]
        init_guess = model.get_init_guess(xx_peaks, yy_peaks, xx, A_init=A_init)
        B_guess = np.nanmin(yy)
        bounds = model.bounds(
            yy, xx, init_guess, peak_center_max_range=peak_center_max_range
        )
        try:
            coeffs_fit, cov = model.fit(
                xx, yy, init_guess, B_guess, bounds=bounds
            )
        except Exception as err:
            traceback.print_exc()
            low, high = bounds
            for i, init_val in enumerate(init_guess):
                low_val = low[i]
                high_val = high[i]
                print(low_val, init_val, high_val, sep=' ; ')
            import pdb; pdb.set_trace()
            
        coeffs_pred = coeffs_fit[:-1]
        B_pred = coeffs_fit[-1] 
        yy_pred = model.model(xx, coeffs_pred, B_pred)
        df_profiles_fit[column] = yy_pred
        RMSE = model.RMSE(yy, yy_pred)
        
        start_idx = 0
        for j in range(n_peaks):
            stop_idx = (j+1)*n_coeffs_per_profile
            coeffs_peak = coeffs_fit[start_idx:stop_idx]
            coeffs_peak = [*coeffs_peak, B_pred, RMSE]
            start_idx = stop_idx
            coeffs_values[(i*n_peaks)+j] = coeffs_peak
        
        if inspect:
            print(f'Fit coefficients = {coeffs_fit}')
            plt.plot(xx, yy_pred)
            plt.scatter(xx, yy)
            plt.show()
            import pdb; pdb.set_trace()
        
        if show_pbar:
            pbar.update()

    index = pd.MultiIndex.from_tuples(
        coeffs_idx, 
        names=['experiment', 'Position_n', 'ID_profile', 'peak_idx']
    )

    df_coeffs = pd.DataFrame(
        index=index,
        columns=['xc_fit', 'sigma_fit', 'A_fit', 'B_fit', 'RMSE'],
        data=coeffs_values
    )
    if show_pbar:
        pbar.close()
        
    return df_profiles_fit, df_coeffs

def radial_distances_nucleolus_nucleus(nucleolus_lab, nucleus_lab, debug=False):
    nucleus_rp = skimage.measure.regionprops(nucleus_lab)
    nucleolus_rp = skimage.measure.regionprops(nucleolus_lab)
    
    nucleolus_mapper = {obj.label: obj for obj in nucleolus_rp}
    
    df_dist = defaultdict(list)
    for obj in nucleus_rp:
        nucleus_contour = get_objContours(
            obj, obj_image=obj.image.max(axis=0).astype(np.uint8)
        )
        rr, cc = nucleus_contour[:,1], nucleus_contour[:,0]
        nucleolus_obj = nucleolus_mapper[obj.label]
        zc, yc, xc = nucleolus_obj.centroid
        for yN, xN in zip(rr, cc):
            yy_line, xx_line = skimage.draw.line(round(yc), round(xc), yN, xN)
            vals = nucleolus_lab[round(zc), yy_line, xx_line]
            change_val_mask = vals[:-1] != vals[1:]
            n_idxs = np.nonzero(change_val_mask)[0]
            yn, xn = yy_line[n_idxs[0]], xx_line[n_idxs[0]]
            
            if debug:
                nucleolus_contour = get_objContours(
                    nucleolus_obj, 
                    obj_image=nucleolus_obj.image.max(axis=0).astype(np.uint8)
                )
                plt.plot(xx_line, yy_line)
                plt.plot(nucleolus_contour[:,0], nucleolus_contour[:,1])
                plt.plot(cc, rr)
                plt.plot([xn, xN, round(xc)], [yn, yN, round(yc)], 'r.')
                plt.show()
                import pdb; pdb.set_trace()
            
            dist = math.dist([yN, xN], [yn, xn])
            df_dist['Cell_ID'].append(obj.label)
            df_dist['nucleolus_to_nucleus_distance'].append(dist)
            
            dist_N = math.dist([yN, xN], [yc, xc])
            df_dist['center_to_edge_nucleus_distance'].append(dist_N)
    
    df = pd.DataFrame(df_dist).set_index('Cell_ID')
    return df
            
        
        
def keep_last_point_less_nans(df_group):
    if 105 in df_group.index and 100 in df_group.index:
        count_nan_105 = df_group.loc[105].isna().sum()
        count_nan_100 = df_group.loc[100].isna().sum()
        drop_idx = 105 if count_nan_105 >= count_nan_100 else 100
        df_group = df_group.drop(index=drop_idx)
        if drop_idx == 100:
            df_group = df_group.rename({105: 100})
    elif 105 in df_group.index:
        df_group = df_group.rename({105: 100})
    return df_group