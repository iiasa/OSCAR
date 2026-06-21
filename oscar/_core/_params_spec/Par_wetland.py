import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst
path_precalib_out = get_paths()["params_precalib"]


##==================
##==================

## parameters for methane wetland emissions
## precalibrated on WETCHIMP simulations 
## (Melton et al., 2013; https://doi.org/10.5194/bg-10-753-2013)
def get_params(mod_region, **useless):

    ## load precalibrated parameters
    with xr.open_dataset(path_precalib_out + f'wetland_WETCHIMP__{mod_region}.nc') as TMP:
        Par = TMP.load()

    ## additional uncertainty factor
    ## note: to span a broader range than (old) WETCHIMP models
    Par.coords['unc_LogNorm'] = ['mean', 'std']
    Par['k_ewet'] = xr.DataArray([[1., 0.2] for _ in range(len(Par.reg_land))], dims=['reg_land', 'unc_LogNorm'], attrs={'units': '1'})

    ## RETURN
    return Par

