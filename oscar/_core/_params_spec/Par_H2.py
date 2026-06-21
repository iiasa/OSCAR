import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## atmospheric H2 parameters and lifetimes 
## based on (Ouyang et al., 2025; https://doi.org/10.1038/s41586-025-09806-1)
def get_params(nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## atmospheric conversion factor
    Par['a_H2'] = xr.DataArray(Cst.M_atm * Cst.m_H2 / Cst.m_air / 1E18, attrs={'units': 'TgH2 ppb-1'})


    ## preindustrial H2 concentration
    ## (Supplementary Note 16)
    ## note: own reconstruction using recent global + firn data + extrapolation
    Par['H2_pi'] = xr.DataArray([319., 24.], dims='unc_LogNorm', attrs={'units': 'ppb'})


    ## present-day (2010-2020) H2 concentration
    ## (Supplementary Note 1)
    Par['H2_pd'] = xr.DataArray(536.2, attrs={'units': 'ppb', 'years': (2010, 2020)})
    M_H2_pd = Par.H2_pd.values * Par.a_H2.values # TgH2


    ## present-day (2010-2020) production factor through methane
    ## (Extended Data Table 1)
    ## note: 517 is the CH4 OH sink obtained over the same period using the OH fields
    ## note: (514 - 31 - 11) / 517 is the paper's scaling factor to match IPCC AR6 CH4 OH sink
    Foh_CH4_pd = 519. * (514 - 31 - 11) / 517 * Cst.m_C / Cst.m_CH4 # Tg C yr-1
    Par['ch_H2_CH4'] = xr.DataArray(26.1 / Foh_CH4_pd.values * np.array([1., np.sqrt((3.5/26.1)**2 - (1.1/9.7)**2)]), dims='unc_LogNorm', attrs={'units': 'TgH2 TgC-1'})


    ## present-day (2010-2020) production factors through VOCs
    ## (Extended Data Table 1) & (Supplementary Notes 4 and 5) & (Supplementary Table 5.1)
    ## note: conversion factor for aggregates assumed the same for all sources (= 0.775)
    ## anthropogenic
    Par['ch_H2_VOC_ant'] = xr.DataArray(1.1 / (107./0.775) * np.array([1., np.sqrt((0.6/1.1)**2 - (22/107.)**2)]), dims='unc_LogNorm', attrs={'units': 'TgH2 Tg-1'})
    ## biomass burning
    Par['ch_H2_VOC_bb'] = xr.DataArray(0.5 / (46.7/0.775) * np.array([1., np.sqrt((0.3/0.5)**2 - (15.3/46.7)**2)]), dims='unc_LogNorm', attrs={'units': 'TgH2 Tg-1'})
    ## biogenic (isoprene + monoterpenes + methanol + others)
    BVOC_bg = 388.7/0.882 + 72.0/0.882 + 40.3/0.375 + 93.3/0.775
    BVOC_unc = np.sqrt((115.7/0.882)**2 + (11.9/0.882)**2 + (5.6/0.375)**2 + (17.9/0.775)**2)
    Par['ch_H2_BVOC'] = xr.DataArray(10.7 / BVOC_bg * np.array([1., np.sqrt((5.0/10.7)**2 - (BVOC_unc/BVOC_bg)**2)]), dims='unc_LogNorm', attrs={'units': 'TgH2 Tg-1'})


    ## present-day (2010-2020) sink rates
    ## (Extended Data Table 1)
    Par['v_H2_OH_pd'] = xr.DataArray(np.array([18.4, 2.2]) / M_H2_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2020)})
    Par['v_H2_soil'] = xr.DataArray(np.array([50.0, 18.0]) / M_H2_pd, dims='unc_LogNorm', attrs={'units': 'yr-1', 'years': (2010, 2020)})


    ## RETURN
    return Par

