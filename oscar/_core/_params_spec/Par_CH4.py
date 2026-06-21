import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## atmospheric CH4 parameters and lifetimes 
## based on AR6 WG1
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## atmospheric conversion factor
    Par['a_CH4'] = xr.DataArray(Cst.M_atm * Cst.m_C / Cst.m_air / 1E18, attrs={'units': 'TgC ppb-1'})


    ## preindustrial CH4 concentration
    ## (Gulev et al., 2021; https://doi.org/10.1017/9781009157896.004) (Section 2.2.3.2.2)
    ## note: min-max over 0-1850 period is 625-807 ppb
    Par['CH4_pi'] = xr.DataArray([729.2, 9.4 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'ppb'})


    ## present-day (2010-2019) CH4 concentration
    ## (Dentener et al., 2021; https://doi.org/10.1017/9781009157896.017) (Table AIII.1a)
    ## note: using our internal conversion factor instead of 2.75 TgCH4 ppb-1 used in Chapter 6
    CH4_pd = np.mean([1798, 1803, 1808, 1814, 1823, 1834, 1842, 1849, 1858, 1866]) # ppb
    M_CH4_pd = CH4_pd * (Par.a_CH4 * Cst.m_CH4 / Cst.m_C).values # TgCH4


    ## present-day (2010-2019) sink rates
    ## (Saunois et al., 2025; https://doi.org/10.5194/essd-17-1873-2025) (Table 3)
    ## note: average of bottom-up and top-down, where possible
    ## note: uncertainty as min-max range, across both where possible, assumed to be 90% range
    Par['v_CH4_OH_pd'] = xr.DataArray(np.array([0.5*(521+602) - 37 - 6 - 0.5*(31+35), 0.5*(663-462) / Cst.s1_to_p90]) / M_CH4_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2019)})
    Par['v_CH4_hv_pd'] = xr.DataArray(np.array([37., 0.5*(43-28) / Cst.s1_to_p90]) / M_CH4_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2019)})
    Par['v_CH4_Cl'] = xr.DataArray(np.array([6., 0.5*(13-1) / Cst.s1_to_p90]) / M_CH4_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2019)})
    Par['v_CH4_soil'] = xr.DataArray(np.array([0.5*(31+35), 0.5*(49-11) / Cst.s1_to_p90]) / M_CH4_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2019)})
    

    ## RETURN
    return Par

