import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_halo import get_params as get_params_halo

path_precalib_out = get_paths()["params_precalib"]

## get list of halogenated compounds
spc_halo = get_params_halo().spc_halo


##==================
##==================

## effective radiative forcing of WMGHGs
## based on AR6 WG1, except SARF precalibrated
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    ## uncertainty
    Par.coords['unc_Norm'] = ['mean', 'std']

    ## list of species
    Par.coords['spc_halo'] = spc_halo


    ## CO2, CH4 and N2O
    ## load precalibrated SARF parameters
    ## (Etminan et al., 2016; https://doi.org/10.1002/2016GL071930)
    with xr.open_dataset(path_precalib_out + f'radiative-forcing_Etminan-2016.nc') as TMP:
        for var in TMP: Par[var] = TMP[var].load()

    ## ERF tropospheric adjustment factors
    ## (Smith et al., 2021; https://www.ipcc.ch/report/ar6/wg1/) (Sections 7.SM.1.3.1 & 7.SM.1.3.2)
    ## note: adding total uncertainty to this factor is arbitrary
    Par['a_adj_CO2'] = xr.DataArray(1.05 * np.array([1., 0.12 / Cst.s1_to_p90]), dims='unc_Norm', attrs={'units': '1'})
    Par['a_adj_CH4'] = xr.DataArray(0.86 * np.array([1., 0.20 / Cst.s1_to_p90]), dims='unc_Norm', attrs={'units': '1'})
    Par['a_adj_N2O'] = xr.DataArray(1.07 * np.array([1., 0.16 / Cst.s1_to_p90]), dims='unc_Norm', attrs={'units': '1'})


    ## halogenated compounds
    ## radiative efficiencies
    ## (WMO, 2022; https://ozone.unep.org/science/assessment/sap) (Table A-5)
    ## note: recommended values including all adjustments
    Par['ph_Xhalo'] = xr.DataArray(np.array([
        0.208, 0.574, 0.200, 0.100, 0.264, 0.276, 0.328, 0.375, 0.415, 0.459, 0.459, 0.515, 0.572, # PFCs
        0.192, 0.111, 0.234, 0.167, 0.169, 0.101, 0.273, 0.263, 0.251, 0.243, 0.359, # HFCs
        0.299, 0.358, 0.279, 0.281, 0.246, 0.302, 0.241, 0.315, 0.297, 0.247, # CFCs
        0.214, 0.068, 0.207, 0.15, 0.161, 0.194, # HCFCs
        4.66E-03, 0.0287, 0.0731, 0.172, 0.0655, # HCCs
        4.21E-03, 0.310, 0.271, 0.309, 0.325 # Halons
        ]) * 1E-3, dims='spc_halo', attrs={'units': 'W m-2 ppt-1'})

    ## ERF uncertainty factor
    ## (Smith et al., 2021; https://www.ipcc.ch/report/ar6/wg1/) (Section 7.SM.1.3.2)
    ## note: arbitrary doubling of uncertainty because applied to each gas separately
    Par['k_ph_halo'] = xr.DataArray([[1., 2*0.19 / Cst.s1_to_p90] for _ in range(len(Par.spc_halo))], dims=['spc_halo', 'unc_Norm'], attrs={'units': '1'})


    ## RETURN
    return Par

