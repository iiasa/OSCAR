import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


##==================
##==================

## effective radiative forcing for albedo effects
def get_params(mod_region, nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_2HalfNorm'] = ['mean', 'std_neg', 'std_pos']


    ## light-absorbing particles on snow and ice
    ## load precalibrated parameters
    ## (Raisanen et al., 2022; https://doi.org/10.5194/acp-22-11579-2022)
    Par_tmp = load_precalib_params('black-carbon_Raisanen-2022', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## LAP adjustments (= efficacy) &  global uncertainty
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Section 7.3.4.3) & (Table 7.8)
    Par['a_adj_lap'] = xr.DataArray(2 * np.array([1, 0.08/0.08 / Cst.s1_to_p90, 0.10/0.08 / Cst.s1_to_p90]), dims='unc_2HalfNorm', attrs={'units': '1'})

    
    ## land cover change albedo
    ## load precalibrated parameters
    ## (Ouyang et al., 2022; https://doi.org/10.1038/s41467-022-31558-z)
    Par_tmp = load_precalib_params('land-albedo_Ouyang-2022', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## global uncertainty factor
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Table 7.8)
    ## note: assumed same as LCC+irrig uncertainty
    Par['k_ERF_lcc'] = xr.DataArray([1, 0.10/0.20 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': '1'})


    ## RETURN
    return Par

