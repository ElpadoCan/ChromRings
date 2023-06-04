import os

import numpy as np
import cv2
from math import sqrt, pow
import skimage.measure
from scipy.optimize import least_squares
from scipy.special import erf

import pandas as pd

import matplotlib.pyplot as plt

from tqdm import tqdm

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
        extra_radius=0, how='object', 
        invert_intensities=True, resample_bin_size_dist=5,
        tqdm_kwargs=None, normalize_every_profile=True,
        normalise_how='max', normalise_average_profile=False, 
        inpsect_single_profiles=False, inpsect_mean_profile=False,
        inner_lab=None, use_absolute_dist=False
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
    inpsect_single_profiles : bool, optional
        Visualise partial results, by default False
    inpsect_mean_profile : bool, optional
        Visualise mean profile, by default False
    inner_lab : np.ndarray, optional
        Optional inner segmentation labels, by default None. 
        If provided, it will be used to determine that 0% of the profile, 
        i.e., the outer edge of the object.
    use_absolute_dist : bool, optional
        Do not use percentage distances, by default False

    Returns
    -------
    list of `skimage.measure.RegionProperties`
        List of region properties. The additional properties computed by this 
        function are `mean_radial_profile`, `radial_profile_argmean`, 
        `radial_profile_argmax`, and `radial_profile_distr_std`.
    """    
    max_dtype = np.iinfo(img_data.dtype).max

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

    for obj in tqdm(rp, desc='Computing radiant profiles', **tqdm_kwargs):
        if inner_lab is None:
            weighted_centroid = obj.weighted_centroid
        else:
            inner_masked = inner_lab[obj.slice][obj.image]
            inner_masked = inner_masked[inner_masked>0]
            inner_ids, inner_count = np.unique(
                inner_masked, return_counts=True
            )
            max_count_idx = inner_count.argmax()    
            inner_id = inner_ids[max_count_idx]
            inner_lab_masked[:] = 0
            inner_lab_masked[inner_lab == inner_id] = obj.label
            inner_obj = skimage.measure.regionprops(inner_lab_masked)[0]
            weighted_centroid = inner_obj.centroid       
        
        if len(weighted_centroid) == 3:
            zc, yc, xc = weighted_centroid
            lab_2D = lab.copy()
            lab_2D[lab_2D != obj.label] = 0
            lab_2D = lab_2D[round(zc)]
            if inner_lab is not None:
                inner_lab_2D = inner_lab_masked[round(zc)]
            obj_z = skimage.measure.regionprops(lab_2D)[0]
            img_data_2D = img_data[round(zc)]
        else:
            yc, xc = weighted_centroid
            obj_z = obj
            lab_2D = lab
            img_data_2D = img_data
            if inner_lab is not None:
                inner_lab_2D = inner_lab_masked
        
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
                    fig, ax = plt.subplots()
                    ax.imshow(inner_contours_lab)
                    ax.plot(xx_line, yy_line)
                    plt.show()
                    import pdb; pdb.set_trace()
                inner_yy, inner_xx = np.where(inner_contours_lab == inner_val)
                inner_y, inner_x = inner_yy[0], inner_xx[0]
                
            # Normalize distance to object edge
            if inner_lab is not None:
                y0, x0 = inner_y, inner_x
            else:
                y0, x0 = y1, x1
            xy_arr = np.column_stack((xx_line, yy_line))
            diff = np.subtract(xy_arr, (x0, y0))
            dist = np.linalg.norm(diff, axis=1)
            zero_dist_idx = dist.argmin()
            if not use_absolute_dist:
                full_dist = sqrt(pow(y2-y0, 2) + pow(x2-x0, 2))
                dist_100 = full_dist - extra_radius
                dist = dist/dist_100
                dist_perc = np.round(dist*100).astype(int)
            else:
                dist_perc = np.round(dist).astype(int)
            
            dist_perc[:zero_dist_idx] *= - 1
            obj.radial_profiles.append(
                {'x': xx_line, 'y': yy_line, 'values': vals, 
                 'norm_dist': dist_perc}
            )

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
                obj.resampled_radial_profiles.append(
                    {'values': resampled['value'].values, 
                     'norm_dist': dist_perc}
                )
                vals = resampled['value'].values

            all_dist.update(dist_perc)

            if inpsect_single_profiles:
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

        obj.mean_radial_profile = obj.radial_df[cols].mean(axis=1)
        obj.stds_radial_profile = obj.radial_df[cols].std(axis=1)
        obj.CVs_radial_profile = (
            (obj.stds_radial_profile/obj.mean_radial_profile)
            .replace(np.inf, np.nan)
        )
        obj.skews_radial_profile = obj.radial_df[cols].skew(axis=1)
        
        obj.mean_radial_profile.name = f'ID_{obj.label}_mean_radial_profile'
        obj.stds_radial_profile.name = f'ID_{obj.label}_CV_radial_profile'
        obj.CVs_radial_profile.name = f'ID_{obj.label}_std_radial_profile'
        obj.skews_radial_profile.name = f'ID_{obj.label}_skew_radial_profile'
        if normalise_average_profile:
            if normalise_how == 'tot_fluo':
                norm_value = img_data[obj.slice][obj.image].sum()
            else:
                norm_func = getattr(np, normalise_how)
                norm_value = norm_func(obj.mean_radial_profile)     
            obj.mean_radial_profile /= norm_value

        profile_series = (
            obj.stds_radial_profile, obj.CVs_radial_profile, 
            obj.skews_radial_profile
        )
        for series_profile in (profile_series):
            norm_func = getattr(np, normalise_how)
            norm_value = norm_func(series_profile)
            series_profile /= norm_value
        
        if inpsect_mean_profile:
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
        obj.radial_profile_argmax = df_norm_distribution.idxmax()/100

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