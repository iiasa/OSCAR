import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


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


    ## global preindustrial baseline (1850-1900)
    ## note: expert judgement (NOAA: 13.7, NASA-GISS: ~13.7, Copernicus: ~13.5, HadCRUT: ~13.5, Berkeley: ~13.8)
    Par['Tg_pi'] = xr.DataArray([13.65 + Cst.degC_to_K.values, 0.4], dims='unc_LogNorm', attrs={'units': '1'})


    ## global temperature and precipitation
    ## load precalibrated parameters
    Par_tmp = load_precalib_params('global-climate_CMIP6')
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## ECS correction to remove feedback from natural emissions
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 6, 7, 8, 9, 10, 11 & 12)
    lambda_NatEm = xr.DataArray(coords={'mod_clim': ['CNRM-ESM2-1', 'UKESM1-0-LL', 'MIROC6', 'NorESM2-LM', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1-G']}, dims=['mod_clim']).fillna(0.)
    lambda_NatEm += [0.0048, -0.0006, -0.016, 0., -0.0006, 0., -0.0077] # dust (AOD)
    lambda_NatEm += [-0.049, 0., -0.015, 0., -0.130, 0., -0.015] # salt (AOD)
    lambda_NatEm += [0., 0.027, 0., 0.0125, 0., 0., -0.0006] # DMS
    lambda_NatEm += [0., 0.001, 0., -0.28, -0.079, -0.084, -0.015] # BVOC (aerosols)
    lambda_NatEm += [0., 0.005, 0., 0., 0.013, 0.014, 0.014] # BVOC (ozone)
    lambda_NatEm += [0., 0.005-0.009, 0., 0., -0.001--0.001, 0.017-0.016, 0.013-0.014] # LNOx (aerosols)
    lambda_NatEm += [0., 0.009, 0., 0., -0.001, 0.016, 0.014] # LNOx (ozone)
    lambda_NatEm += [0., -0.079, 0., 0., -0.062, 0., -0.050] # climate (ozone)
    Par['lambda_0'] = Par.lambda_0 - lambda_NatEm.combine_first(0 * Par.lambda_0)


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
    Par_tmp = load_precalib_params('regional-climate_CMIP6', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## keep only preindustrial local precipitation for relative change
    Par = Par.drop_vars(['Tg_piC', 'Tl_piC', 'Pg_piC'])

    ## no noise for preindustrial local precipitation
    Par['Pl_piC'].attrs['mod_noise_override'] = 0.


    ## RETURN
    return Par


