import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst

path_precalib_out = get_paths()["params_precalib"]


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
    with xr.open_dataset(path_precalib_out + f'land_TRENDY__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## additional uncertainty factor
    ## note: to span a broader range than TRENDY models
    Par['k_npp'] = xr.DataArray([[[1., 0.2] for _ in range(len(Par.bio_land))] for _ in range(len(Par.reg_land))], dims=['reg_land', 'bio_land', 'unc_LogNorm'], attrs={'units': '1'})

    ## no noise for fraction parameters
    Par['p_harv'].attrs['mod_noise_override'] = 0.
    Par['p_graz'].attrs['mod_noise_override'] = 0.

    ## use Dirichlet distribution for fraction parameters
    Par = Par.rename({'mod_Eharv': 'dir_Eharv', 'mod_Egraz': 'dir_Egraz'})


    ## land C cycle (subpool partitioning)
    ## load precalibrated parameters
    with xr.open_dataset(path_precalib_out + f'land_ISIMIP3a__{mod_region}.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## no noise for fraction parameters
    Par['p_wood'].attrs['mod_noise_override'] = 0.
    Par['p_root'].attrs['mod_noise_override'] = 0.
    Par['p_leaf'].attrs['mod_noise_override'] = 0.
    Par['p_litter'].attrs['mod_noise_override'] = 0.

    ## use Dirichlet distribution for fraction parameters
    Par = Par.rename({'mod_Cveg_part': 'dir_Cveg_part', 'mod_Csoil_part': 'dir_Csoil_part'})


    ## coarse woody debris decay
    ## (Harmon et al., 2020; https://doi.org/10.1186/s13021-019-0136-6) (Table 1)
    Par['v10_cwd'] = xr.DataArray([0.061, 0.006], dims='unc_LogNorm', attrs={'units': '1'})
    Par['q10_cwd'] = xr.DataArray([2.50, 0.20], dims='unc_LogNorm', attrs={'units': '1'})

    ## respiration fraction of CWD
    ## (Stokland et al., 2024; https://doi.org/10.1007/s10533-024-01170-y) (Conclusion)
    ## note: no real data for tropics, assumed similar
    ## note: arbitrary uncertainty
    Par['p_cwd_resp'] = xr.DataArray([0.6, 0.1], dims='unc_LogitNorm', attrs={'units': '1'})


    ## litterfall as fraction of NPP
    ## (Neumann et al., 2018; https://doi.org/10.1029/2017GB005825) (Section 4)
    ## (Chave et al., 2010; https://doi.org/10.5194/bg-7-43-2010) (Introduction)
    ## note: no clean source but around 1/3 of NPP seems robust across literature
    ## note: arbitrary uncertainty
    Par['p_fall_npp'] = xr.DataArray([0.35, 0.05], dims='unc_LogitNorm', attrs={'units': '1'})


    ## RETURN
    return Par

