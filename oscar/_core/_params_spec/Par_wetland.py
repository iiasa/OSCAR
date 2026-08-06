import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


##==================
##==================

## parameters for methane wetland emissions
## precalibrated on WETCHIMP simulations 
## (Melton et al., 2013; https://doi.org/10.5194/bg-10-753-2013)
def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()

    ## load precalibrated parameters
    Par_tmp = load_precalib_params('wetland_WETCHIMP', mod_region, xxx_global=True, xxx_global_ignore=['Awet_piW'])
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## preindustrial climate taken from another module
    Par = Par.drop_vars(['Tl_piW', 'Pl_piW'])

    ## additional uncertainty factor
    ## note: to span a broader range than (old) WETCHIMP models
    Par.coords['unc_LogNorm'] = ['mean', 'std']
    Par['k_ewet'] = xr.DataArray([[1., 0.2] for _ in range(len(Par.reg_land))], dims=['reg_land', 'unc_LogNorm'], attrs={'units': '1'})
    #Par['k_ewet'] = xr.DataArray(1., attrs={'units': '1'})

    ## RETURN
    return Par

