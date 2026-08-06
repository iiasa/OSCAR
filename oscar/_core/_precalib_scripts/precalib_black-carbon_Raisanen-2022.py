import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'black-carbon_Raisanen-2022'


##################################################
##   PRECALIBRATION OF LAP ON SNOW
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data on selected regions
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ds = aggreg_regions(TMP, mod_region)
            ds = ds.compute()

        ## initialization
        Par = xr.Dataset()

        ## regional parameters
        Par['ph_lap_BC'] = ds.RFsnow_BC / ds.E_BC
        Par['ph_lap_BC'].attrs['units'] = 'W yr m-2 TgC-1' 

        ## global parameters
        Par['ph_lap_BC_glb'] = ds.RFsnow_BC.sum('reg_land', min_count=1) / ds.E_BC.sum('reg_land', min_count=1)
        Par['ph_lap_BC_glb'].attrs['units'] = 'W yr m-2 TgC-1' 

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

