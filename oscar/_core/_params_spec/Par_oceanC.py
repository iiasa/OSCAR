import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## ocean structure and impulse response function
## (Strassmann & Joos, 2018; https://doi.org/10.5194/gmd-11-1887-2018) (Tables A2 & A3)
## (Joos et al., 1996; https://doi.org/10.3402/tellusb.v48i3.15921) (Appendix A.2.4)
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['mod_Focean_struct'] = ['HILDA', 'Bern-2.5D', 'Princeton-GCM']
    Par.coords['box_surf'] = np.arange(5)
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## DIC conversion factor
    Par['a_dic'] = xr.DataArray(1 / 1026.5 / 12.0107E-6 * 1E15, attrs={'units': 'umol m3 PgC-1 kgSW-1'})

    ## preindustrial mixing layer depth
    mld_0 = xr.DataArray([75., 50.0, 50.9], dims='mod_Focean_struct', attrs={'units': 'm'})

    ## global ocean area
    A_surf = xr.DataArray(1E14 * np.array([3.62, 3.5375, 3.55]), dims='mod_Focean_struct', attrs={'units': 'm2'})

    ## surface layer volume
    Par['V_mld'] = mld_0 * A_surf
    Par['V_mld'].attrs['units'] = 'm3'
    
    ## preindustrial global ocean temperature
    Par['To_0'] = xr.DataArray(np.array([18.17, 18.30, 17.70]), dims='mod_Focean_struct', attrs={'units': 'degC'})

    ## gaseous exchange speed at ocean surface
    Par['v_fg'] = xr.DataArray(1 / np.array([9.06, 7.46, 7.66]), dims='mod_Focean_struct', attrs={'units': 'yr-1'})

    ## time-scales of oceanic surface-to-deep transport
    Par['t_circ'] = xr.DataArray([
        [2.1990, 12.038, 59.584, 237.31, 1E18], 
        [2.6900, 13.617, 86.797, 337.30, 1E18], 
        #[2.0090, 16.676, 65.102, 347.58, 1E18]], # this is from (Strassmann & Joos, 2018)
        [2.3488, 15.281, 65.359, 347.55, 1E18]], # original values from (Joos et al., 1996)
        dims=['mod_Focean_struct', 'box_surf'], attrs={'units': 'yr'})

    ## fraction of ocean surface boxes
    Par['p_surf'] = xr.DataArray([
        [0.23337, 0.13733, 0.051541, 0.035033, 0.022936], 
        [0.094671, 0.10292, 0.0392835, 0.012986, 0.013691], 
        #[1.2817, 0.061618, 0.037265, 0.019565, 0.014818]], # this is from (Strassmann & Joos, 2018)
        [0.24966, 0.066485, 0.038344, 0.019439, 0.014819]], # original values from (Joos et al., 1996)
        dims=['mod_Focean_struct', 'box_surf'], attrs={'units': '1', 'mod_noise_override': 0.})


    ## additional parameters to increase model spread
    ## CMIP6-based relative uncertainty in beta/gamma ocean sensitivities 
    ## (Arora et al., 2020; https://doi.org/10.5194/bg-17-4173-2020) (Abstract)
    Par['k_b_dic'] = xr.DataArray([1, 0.07/0.79], dims='unc_LogNorm', attrs={'units': '1'})
    Par['k_g_dic'] = xr.DataArray([1, 5/17.2], dims='unc_LogNorm', attrs={'units': '1'})


    ## RETURN
    return Par

