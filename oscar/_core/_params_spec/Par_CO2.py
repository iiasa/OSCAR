import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## atmospheric CO2 parameters
## based on AR6 WG1
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## atmospheric conversion factor
    Par['a_CO2'] = xr.DataArray(Cst.M_atm * Cst.m_C / Cst.m_air / 1E18, attrs={'units': 'PgC ppm-1'})


    ## preindustrial CO2 concentration
    ## (Gulev et al., 2021; https://doi.org/10.1017/9781009157896.004) (Section 2.2.3.2.1)
    ## note: min-max over 0-1850 period is 274-285 ppm
    Par['CO2_pi'] = xr.DataArray([278.2, 2.9 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'ppm'})


    ## RETURN
    return Par

