import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


##==================
##==================

## parameters for biomass burning emissions
## precalibrated on GFED4 data
## (van der Werf et al., 2017; https://doi.org/10.5194/essd-9-697-2017)
def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()

    ## load precalibrated parameters
    Par_tmp = load_precalib_params('biomass-burning_GFED4', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## RETURN
    return Par

