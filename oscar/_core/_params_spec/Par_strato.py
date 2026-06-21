import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_halo import get_params as get_params_halo


## get halogenated compounds parameters
Par_halo = get_params_halo()


##==================
##==================

## parameters for stratospheric chemistry
## compiled from various sources
def get_params(nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## list of ODSs
    Par.coords['spc_halo'] = ['CFC-11', 'CFC-12', 'CFC-113', 'CFC-114', 'CFC-115', 
        'HCFC-22', 'HCFC-141b', 'HCFC-142b',
        'CH3CCl3', 'CCl4', 'CH3Cl', 'CH2Cl2', 'CHCl3', 
        'CH3Br', 'Halon-1211', 'Halon-1202', 'Halon-1301', 'Halon-2402']


    ## mean age of air for lagged concentration
    Par.coords['age_air'] = ['mid_lat', 'high_lat']
    Par['t_lag'] = xr.DataArray([3., 5.5], dims='age_air', attrs={'units': 'yr'})


    ## EESC parameters
    ## TODO: add uncertainty? (https://doi.org/10.5194/acp-14-2757-2014) (Table 2)
    ## time-independent fractional release factors
    ## (Engel et al., 2018; https://doi.org/10.5194/acp-18-601-2018) (Tables 1 and 2)
    Par['p_fracrel'] = xr.DataArray(
        [[0.47, 0.24, 0.30, 0.13, 0.07, 
        0.15, 0.34, 0.17, 
        0.61, 0.56, 0.44, np.nan, np.nan, 
        0.60, 0.65, 0.67, 0.32, 0.66], 
        [0.99, 0.87, 0.91, 0.41, 0.20, 
        0.44, 0.90, 0.65,
        0.99, 1.00, 0.91, np.nan, np.nan, 
        0.99, 1.00, 1.00, 0.83, 1.00]],
        dims=['age_air', 'spc_halo'], attrs={'units': '1', 'mod_noise_override': 0.})


    ## relative efficency of Bromine vs. Chlorine
    ## (Engel et al., 2018; https://doi.org/10.5194/acp-18-601-2018)
    Par['a_Cl_Br'] = xr.DataArray(60., attrs={'units': '1'})


    ## hv lifetime sensitivities
    ## TODO: add solar?
    ## (Prather et al., 2005; https://doi.org/10.1002/2015JD023267)
    ## note: taking opposite values because sensivity of v = 1/tau

    ## sensitivity to N2O
    ## (Table 2)
    Par['ch_hv_N2O'] = xr.DataArray([0.065, 0.010], dims='unc_LogNorm', attrs={'units': '1'})

    ## sensitivity to ODSs
    ## note: based on the 3% to 5% change mentioned in text, we assume mean = 4% and std = 1%
    ## note: normalized with:
    ## 1) AR5 ODS concentrations (interpolated over 2002-2007, as 2005-2010 minus 3 years) 
    ## (Prather et al., 2013; https://doi.org/10.1017/CBO9781107415324.030) (Table AII.1.1b)
    ## 2) corresponding preindustrial ODS concentrations (from CMIP5/MAGICC)
    ## (Meinshausen et al., 2011; https://doi.org/10.1007/s10584-011-0156-z) (Table 1)
    ## 3) chosen fractional release factors
    ## 4) EESC rounded to 10s of ppt (1780 and 480 ppt with Engel's, 1750 and 420 ppt with Newman's)
    ODS_2000 = np.array([261.7, 541, 82.3, 16.5, 7.9, 139.5, 11.8, 11.4, 49.7, 98.6, 550, np.nan, np.nan, 8.9, 4.02, 0.04, 2.84, 0.5]) # ppt
    ODS_2005 = np.array([251.6, 542.7, 78.8, 16.6, 8.3, 165.5, 17.5, 15.1, 20.1, 93.7, 550, np.nan, np.nan, 7.9, 4.26, 0.02, 3.03, 0.48]) # ppt
    ODS_2010 = np.array([240.9, 532.5, 75.6, 16.4, 8.4, 206.8, 20.3, 20.5, 8.3, 87.6, 550, np.nan, np.nan, 7.2, 4.07, 0, 3.2, 0.46]) # ppt
    ODS_pd = (1.2 * ODS_2000 + 4.2 * ODS_2005 + 0.6 * ODS_2010) / 6.
    EESC_pd = ((Par.p_fracrel * (Par_halo.n_Cl + Par.a_Cl_Br * Par_halo.n_Br)).sel(age_air='mid_lat').values * ODS_pd).sum(where=~np.isnan(ODS_pd)).round(-1)
    EESC_pi = ((Par.p_fracrel * (Par_halo.n_Cl + Par.a_Cl_Br * Par_halo.n_Br)).sel(spc_halo=['CH3Cl', 'CH3Br'], age_air='mid_lat').values * np.array([480., 5.8])).sum().round(-1)
    Par['ch_hv_EESC'] = xr.DataArray(np.array([0.04, 0.01]) / np.log(EESC_pd / EESC_pi), dims='unc_LogNorm', attrs={'units': '1'})

    ## sensitivity to stratospheric circulation
    ## note: assumed to match total 6% change mentioned in text, with 100% uncertainty to include G2d model, giving mean = 0.5% and std = 0.5%
    ## note: normalized with rounded mean age of air from G2d model
    ## (Fleming et al., 2011; https://doi.org/10.5194/acp-11-8515-2011) (Figure 12)
    ageair_pd, ageair_pi = 4.0, 4.5 # yr
    Par['ch_hv_ageair'] = xr.DataArray(np.array([-0.005, 0.005]) / -np.log(ageair_pd / ageair_pi), dims='unc_LogNorm', attrs={'units': '1'})


    ## climate-related parameters
    ## note: calibrated for OSCARv2 based on CCMVal2 data
    ## (Morgenstern et al., 2010; https://doi.org/10.1029/2009JD013728)
    Par.coords['mod_ageair'] = ['AMTRAC', 'CAM3.5', 'CMAM', 'Niwa-SOCOL', 'SOCOL', 'ULAQ', 'UMUKCA-UCAM']
    Par['g_ageair'] = xr.DataArray([0.15, 0.091375, 0.090625, 0.0683125, 0.1441875, 0.11775, 0.086125], dims='mod_ageair', attrs={'units': 'K-1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['g_ageair'] = xr.DataArray([0.107, 0.029], dims='unc_LogNorm', attrs={'units': 'K-1'})
        Par = Par.drop_dims('mod_ageair')


    ## RETURN
    return Par


##==================
##==================

'''
## FYI: old factors based on Newman et al. (taken from Engel's tables)
## (Newman et al., 2007; https://doi.org/10.5194/acp-7-4537-2007)
p_fracrel_old = xr.DataArray(
    [[0.47, 0.23, 0.29, 0.12, 0.04, 
    0.13, 0.34, 0.17,
    0.67, 0.56, 0.44, np.nan, np.nan, 
    0.60, 0.62, 0.62, 0.28, 0.65], 
    [0.99, 0.86, 0.90, 0.40, 0.15, 
    0.41, 0.90, 0.65, 
    0.99, 1.00, 0.91, np.nan, np.nan, 
    0.99, 1.00, 1.00, 0.80, 1.00]],
    dims=['age_air', 'spc_halo'], attrs={'units': '1'})
'''

