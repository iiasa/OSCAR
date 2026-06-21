import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst

path_precalib_out = get_paths()["params_precalib"]


##==================
##==================

## default driver-related parameters
## note: should be replaced with setup-specific values whenever possible!
def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()


    ## load precalibrated parameters from CMIP7
    with xr.open_dataset(path_precalib_out + f'drivers-params_CMIP7__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()


    ## load precalibrated parameters from IGCC
    ## note: this replaces the CMIP7 concentrations
    with xr.open_dataset(path_precalib_out + f'drivers-params_IGCC.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()


    ## fraction of geological anthropogenic CH4 emissions
    ## note: zero means CO2 emissions include oxidized CH4, e.g. because estimated through C content
    sect_geo = ['air', 'dom', 'ene', 'ind', 'shp', 'slv', 'tra']
    Par['p_Egeo_CH4'] = 0.


    ## RETURN
    return Par

