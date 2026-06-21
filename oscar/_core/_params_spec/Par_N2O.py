import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## atmospheric N2O parameters and lifetimes 
## based on AR6 WG1
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## atmospheric conversion factor
    Par['a_N2O'] = xr.DataArray(Cst.M_atm * 2*Cst.m_N / Cst.m_air / 1E18, attrs={'units': 'TgN ppb-1'})


    ## preindustrial N2O concentration
    ## (Gulev et al., 2021; https://doi.org/10.1017/9781009157896.004) (Section 2.2.3.2.3)
    ## note: min over 0-1850 period at ~261 ppb
    Par['N2O_pi'] = xr.DataArray([270.1, 6.0 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'ppb'})


    ## present-day (2010-2019) N2O concentration
    ## (Dentener et al., 2021; https://doi.org/10.1017/9781009157896.017) (Table AIII.1a)
    ## using our internal conversion factor
    N2O_pd = np.mean([323.4, 324.4, 325.3, 326.2, 327.4, 328.3, 329.1, 330.0, 331.2, 332.1]) # ppb
    M_N2O_pd = N2O_pd * Par.a_N2O.values # TgN


    ## present-day (2010-2019) lifetime
    ## (Hanqin et al., 2024; https://doi.org/10.5194/essd-16-2543-2024) (Table 3)
    ## note: average of bottom-up and top-down, 
    ## note: uncertainty as min-max range across both, not reduced because fairly low already!
    Par['v_N2O_hv_pd'] = xr.DataArray(np.array([0.5*(13.4+12.6), 0.5*(14.5-12.3)]) / M_N2O_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2019)})


    ## RETURN
    return Par

