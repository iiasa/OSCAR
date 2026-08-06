import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._base.fct_load import load_precalib_params


##==================
##==================


def get_params(mod_region, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_LogNorm'] = ['mean', 'std']
    Par.coords['unc_Triangle'] = ['mode', 'mini', 'maxi']
    Par.coords['box_hwp'] = ['fuel', 'paper', 'solid']
    
   
    ## harvested wood products
    ## load precalibrated parameters
    ## (Earles et al., 2022; https://doi.org/10.1038/nclimate1535)
    Par_tmp = load_precalib_params('wood-use_Earles-2012', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## fraction of non-commercial HWP used as fuel
    Par['p_fuel_noncomm'] = xr.DataArray([[0.5, 0., 1.] for _ in range(len(Par.reg_land))], dims=['reg_land', 'unc_Triangle'], attrs={'units': '1'})

    ## turnover rate of HWP pools
    ## (IPCC, 2006; https://www.ipcc-nggip.iges.or.jp/public/2006gl/vol4.html) (Chapter 12, Table 12.2)
    ## note: arbitrary value for 'fuel' category
    Par['v_hwp'] = xr.DataArray(np.log(2) / np.array([0.5, 2, 30]), dims='box_hwp', attrs={'units': 'yr-1'})


    ## rotation period for shifting cultivation 
    ## (Hurtt et al., 2006; https://doi.org/10.1111/j.1365-2486.2006.01150.x)
    Par['t_shift'] = xr.DataArray(15., attrs={'units': 'yr'})


    ## RETURN
    return Par

