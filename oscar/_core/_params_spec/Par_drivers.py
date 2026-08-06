import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


##==================
##==================

## default driver-related parameters
## note: should be replaced with setup-specific values whenever possible!
def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()

    ## load precalibrated parameters from CMIP7
    Par_tmp = load_precalib_params('drivers-params_CMIP7', mod_region, xxx_global=False)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## load precalibrated parameters from IGCC
    ## note: this replaces the CMIP7 concentrations
    Par_tmp = load_precalib_params('drivers-params_IGCC')
    Par = xr.merge([Par, Par_tmp], join='outer', compat='override')

    ## RETURN
    return Par

