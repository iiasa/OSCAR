import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst

path_precalib_out = get_paths()["params_precalib"]

##==================
##==================

## temperature and precipitation response
## based on CMIP6 models
def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogitNorm'] = ['mean', 'std']


    ## global temperature and precipitation
    ## load precalibrated parameters
    with xr.open_dataset(path_precalib_out + f'global-climate_CMIP6.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## fraction of energy warming up the ocean
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Table 7.1)
    ## note: arbitrary uncertainty
    Par['p_ohc'] = xr.DataArray([0.91, 0.01], dims='unc_LogitNorm', attrs={'units': '1'})


    ## atmospheric fractions of radiative forcing
    ## (Andrews et al., 2010; https://doi.org/10.1029/2010GL043991) (Table 3)
    ## (Kvalevag et al., 2013; https://doi.org/10.1002/grl.50318) (Table 2)
    ## note: rounded values, average excl. very uncertain exp. from Kvalevag et al., some arbitrary uncertainties
    Par['p_atm_CO2'] = xr.DataArray([0.7, 0.1], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_nonCO2'] = xr.DataArray([0.5, 0.2], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_O3'] = xr.DataArray([-0.3, 0.2], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_scatter'] = xr.DataArray([-0.2, 0.2], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_absorb'] = xr.DataArray([4.3, 1.8], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_alb'] = xr.DataArray([0.0, 0.1], dims='unc_Norm', attrs={'units':'1'})
    Par['p_atm_solar'] = xr.DataArray([0.2, 0.1], dims='unc_Norm', attrs={'units':'1'})


    ## regional temperature and precipitation
    ## load precalibrated parameters
    with xr.open_dataset(path_precalib_out + f'regional-climate_CMIP6__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()


    ## RETURN
    return Par


