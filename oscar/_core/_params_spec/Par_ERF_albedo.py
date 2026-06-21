import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst
path_precalib_out = get_paths()["params_precalib"]


##==================
##==================

## effective radiative forcing for SCLFs
## based on AR6 WG1 and CMIP6/AerchemMIP
def get_params(mod_region, nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_2HalfNorm'] = ['mean', 'std_neg', 'std_pos']


    ## light-absorbing particles on snow and ice
    ## load precalibrated parameters
    ## (Raisanen et al., 2022; https://doi.org/10.5194/acp-22-11579-2022)
    with xr.open_dataset(path_precalib_out + f'black-carbon_Raisanen-2022__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## LAP adjustments (= efficacy) &  global uncertainty
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Section 7.3.4.3) & (Table 7.8)
    Par['a_adj_lap'] = xr.DataArray(2 * np.array([1, 0.08/0.08 / Cst.s1_to_p90, 0.10/0.08 / Cst.s1_to_p90]), dims='unc_2HalfNorm', attrs={'units': '1'})

    
    ## land cover change albedo
    ## load precalibrated parameters
    ## (Ouyang et al., 2022; https://doi.org/10.1038/s41467-022-31558-z)
    with xr.open_dataset(path_precalib_out + f'land-albedo_Ouyang-2022__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## global uncertainty factor
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Table 7.8)
    ## note: assumed same as LCC+irrig uncertainty
    Par['k_ERF_lcc'] = xr.DataArray([1, 0.10/0.20 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': '1'})


    ## RETURN
    return Par

