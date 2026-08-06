import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'wood-use_Earles-2012'


##################################################
##   PRECALIBRATION OF WOOD USE ALLOCATION
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ## regional
            ds = aggreg_regions(TMP, mod_region, weight_var={var: '_area' for var in TMP if var != '_area'})
            ds = ds.drop_vars('_area').compute()
            ## global
            ds_glob = (TMP * TMP._area).sum('reg_code', min_count=1) / TMP._area.sum('reg_code', min_count=1)
            ds_glob = ds_glob.drop_vars('_area').compute()

        ## ASSUMPTION:
        ## gap-fill with global average
        ds = ds.where(ds.notnull(), ds_glob)

        ## initialization
        Par = xr.Dataset()

        ## wood use partition
        Par['p_hwp_comm'] = xr.concat([ds[var].assign_coords(box_hwp=var).expand_dims('box_hwp', -1) for var in ['fuel', 'paper', 'solid']], dim='box_hwp')
        Par['p_hwp_noncomm'] = xr.concat([(var == 'fuel') * ds['non_commercial'].assign_coords(box_hwp=var).expand_dims('box_hwp', -1) for var in ['fuel', 'paper', 'solid']], dim='box_hwp')
        p_sum = Par.p_hwp_comm.sum('box_hwp', min_count=1) + Par.p_hwp_noncomm.sum('box_hwp', min_count=1)
        Par['p_hwp_comm'] /= p_sum
        Par['p_hwp_noncomm'] /= p_sum

         ## add units
        for var in ['p_hwp_comm', 'p_hwp_noncomm']:
            Par[var].attrs['units'] = '1'

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

