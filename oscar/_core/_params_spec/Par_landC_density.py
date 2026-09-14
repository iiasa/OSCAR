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
    Par.coords['unc_LogitNorm'] = ['mean', 'std']


    ## land C cycle (preindustrial and sensitivities)
    ## load precalibrated parameters
    Par_tmp = load_precalib_params('land_TRENDY', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## take these as true preindustrial climate
    Par = Par.rename({'Tl_piL': 'Tl_pi', 'Pl_piL': 'Pl_pi'})

    ## set zero fire in Cropland and Urban biomes
    Par['v_fire_piL'].loc[{'bio_land': ['Cropland', 'Urban']}] = 0.

    ## assume no change in v_mort and v_resp
    ## note: because only direct CO2 factors are considered to re-align preindustrial
    Par = Par.rename({'v_mort_piL': 'v_mort', 'v_resp_piL': 'v_resp'})

    ## additional uncertainty factor
    ## note: to span a broader range than TRENDY models
    Par['k_npp'] = xr.DataArray([[[1., 0.2] for _ in range(len(Par.bio_land))] for _ in range(len(Par.reg_land))], dims=['reg_land', 'bio_land', 'unc_LogNorm'], attrs={'units': '1'})
    #Par['k_npp'] = xr.DataArray(1., attrs={'units': '1'})

    ## no noise for fraction parameters
    Par['p_charv'].attrs['mod_noise_override'] = 0.
    Par['p_graz'].attrs['mod_noise_override'] = 0.

    ## use Dirichlet distribution for fraction parameters
    Par = Par.rename({'mod_Echarv': 'dir_Echarv', 'mod_Egraz': 'dir_Egraz'})


    ## land C cycle (subpool partitioning)
    ## load precalibrated parameters
    Par_tmp = load_precalib_params('land_ISIMIP3a', mod_region, xxx_global=True)
    Par = xr.merge([Par, Par_tmp], join='outer', compat='no_conflicts')

    ## no noise for fraction parameters
    Par['p_wood'].attrs['mod_noise_override'] = 0.
    Par['p_root'].attrs['mod_noise_override'] = 0.
    Par['p_leaf'].attrs['mod_noise_override'] = 0.
    Par['p_litter'].attrs['mod_noise_override'] = 0.

    ## use Dirichlet distribution for fraction parameters
    Par = Par.rename({'mod_Cveg_part': 'dir_Cveg_part', 'mod_Csoil_part': 'dir_Csoil_part'})


    ## coarse woody debris decay
    ## (Harmon et al., 2020; https://doi.org/10.1186/s13021-019-0136-6) (Table 1)
    Par['v10_cwd'] = xr.DataArray([0.061, 0.006], dims='unc_LogNorm', attrs={'units': 'yr-1'})
    Par['q10_cwd'] = xr.DataArray([2.50, 0.20], dims='unc_LogNorm', attrs={'units': '1'})

    ## respiration fraction of CWD
    ## (Stokland et al., 2024; https://doi.org/10.1007/s10533-024-01170-y) (Conclusion)
    ## note: no real data for tropics, assumed similar
    ## note: arbitrary uncertainty
    Par['p_cwd_resp'] = xr.DataArray([[[0.6, 0.1] for _ in range(len(Par.bio_land))] for _ in range(len(Par.reg_land))], dims=['reg_land', 'bio_land', 'unc_LogitNorm'], attrs={'units': '1'})


    '''
    ## fraction of npp going to woody biomass
    ## (Xia et al., 2019; https://doi.org/10.1029/2018JG004777) (Figure 2)
    ## (Lu et al., 2025; https://doi.org/10.1111/jbi.15094) (Supplementary Information)
    ## (Malhi et al.https://doi.org/10.1098/rstb.2011.0062) (Abstract)
    ## note: rounded value, uncertainty from third study
    Par['p_npp_wood'] = xr.DataArray([0.4, 0.1], dims=['unc_LogitNorm'], attrs={'units': '1'})
    '''

    ## turnover time of woody biomass for CWD production
    ## (Yu et al., 2023; https://doi.org/10.1111/geb.13736) [63-99 yr] (raw data, not ML extrapolation)
    ## (Xue et al., 2017; https://doi.org/10.1002/2016GB005557) [67; 56-104 yr] (using productivity)
    ## (Galbraith et al., 2013; https://doi.org/10.1080/17550874.2013.770578) [50; 23-129 yr] (tropics, using productivity)
    ## (Lewis et al., 2004; https://doi.org/10.1111/j.0022-0477.2004.00923.x) [55 yr] (tropics, stem demographics)
    ## note: arbitrary but informed by above refs, can be changed to adjust CWD pool
    Par['t_wood'] = xr.DataArray([55., 20.], dims=['unc_LogNorm'], attrs={'units': 'yr'})


    ## RETURN
    return Par

