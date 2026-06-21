import numpy as np
import xarray as xr

from scipy.optimize import fsolve

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## effective radiative forcing for SCLFs
## based on AR6 WG1 and CMIP6/AerchemMIP
def get_params(nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## reference CH4 concentrations
    ## (Dentener et al., 2021; https://doi.org/10.1017/9781009157896.017) (Table AIII.1a)
    CH4_pd, CH4_pi = 1866., 729. # ppb

    ## CH4-induced SWV and O3 shape parameters
    ## (Winterstein et al., 2019; https://doi.org/10.5194/acp-19-7151-2019) (Table 1)
    Par['x_rf_swv'] = xr.DataArray(fsolve(lambda b: (5**b -1) / (2**b - 1) - 0.55/0.15, x0=1.)[0], attrs={'units': '1'})
    Par['x_rf_O3'] = xr.DataArray(fsolve(lambda b: (5**b -1) / (2**b - 1) - 0.76/0.27, x0=1.)[0], attrs={'units': '1'})


    ## stratospheric water vapor (SWV) sensitivity
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Table 7.8)
    nonlin_swv = np.atleast_1d(Par.x_rf_swv / ((CH4_pd / CH4_pi) ** Par.x_rf_swv - 1))
    Par['Ph_swv_CH4'] = xr.DataArray(0.050 * nonlin_swv * np.array([1., 1.00 / Cst.s1_to_p90]), dims='unc_LogNorm', attrs={'units': 'W m-2'})


    ## O3 radiative+chemical (linear) sensitivities
    ## (Forster et al., 2021; https://doi.org/10.1017/9781009157896.009) (Section 7.SM.1.4) & (Table 7.SM.3)
    ## note: uncertainty for N2O corrected (see: https://github.com/chrisroadmap/ar6/blob/main/notebooks/070_chapter7_ozone_emissions_to_forcing.ipynb)
    ## note: LogNorm distrib when sign agreement in original data and Norm otherwise
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-853-2021) (Table S7)
    ph_O3_CH4 = xr.DataArray([0.175E-3, 0.062E-3 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_O3_N2O'] = xr.DataArray([0.710E-3, 0.471E-3 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_O3_EESC'] = xr.DataArray([-0.125E-3, 0.113E-3 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'W m-2 ppt-1'})
    Par['ph_O3_CO'] = xr.DataArray(np.array([0.155E-3, 0.131E-3 / Cst.s1_to_p90]) * (Cst.m_CO/Cst.m_C).values, dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgC-1'})
    Par['ph_O3_VOC'] = xr.DataArray([0.329E-3, 0.328E-3 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'W yr m-2 Tg-1'})
    Par['ph_O3_NOx'] = xr.DataArray(np.array([1.797E-3, 0.983E-3 / Cst.s1_to_p90]) * (Cst.m_NO2/Cst.m_N).values, dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgN-1'})

    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 10, 11 & 14)
    Par['ph_O3_BVOC'] = xr.DataArray([1.3E-4, 0.4E-4], dims='unc_LogNorm', attrs={'units': 'W yr m-2 Tg-1'})
    Par['ph_O3_LNOx'] = xr.DataArray([0.034, 0.009], dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgN-1'})
    Par['ph_O3_Tg'] = xr.DataArray([-0.064, 0.012], dims='unc_LogNorm', attrs={'units': 'W m-2 K-1'})

    ## reformulation of O3-CH4 sensitivity to match shape parameter
    nonlin_O3 = np.atleast_1d(Par.x_rf_O3 / ((CH4_pd / CH4_pi) ** Par.x_rf_O3 - 1))
    Par['Ph_O3_CH4'] = xr.DataArray(ph_O3_CH4 * (CH4_pd - CH4_pi) * nonlin_O3, attrs={'units': 'W m-2'})

    ## ERF tropospheric adjustment factor for O3
    ## note: as in AR6, assuming SARF = ERF
    Par['a_adj_O3'] = xr.DataArray(1., attrs={'units': '1'})


    ## aerosol-radiation interactions (linear) sensitivities
    ## anthropogenic factors
    ## (Smith et al., 2024; https://doi.org/10.5194/gmd-17-8569-2024) (Table 3)
    ## note: LogNorm distrib when sign agreement in original data and Norm otherwise, except secondary drivers use Norm
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-853-2021) (Tables S2 & S8)
    Par['ph_ari_BC'] = xr.DataArray([2.79E-2, 2.39E-2 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W yr m-2 TgC-1'})
    Par['ph_ari_OC'] = xr.DataArray([-4.33E-3, 3.06E-3 / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgC-1'})
    Par['ph_ari_SO2'] = xr.DataArray(np.array([-3.08E-3, 2.29E-3 / Cst.s1_to_p90]) * (Cst.m_SO2/Cst.m_S).values, dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgS-1'})
    Par['ph_ari_NH3'] = xr.DataArray(np.array([-6.21E-4, 6.90E-5 / Cst.s1_to_p90]) * (Cst.m_NH3/Cst.m_N).values, dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgN-1'})
    Par['ph_ari_NOx'] = xr.DataArray(np.array([-8.17E-5, 3.15E-5 / Cst.s1_to_p90]) * (Cst.m_NO2/Cst.m_N).values, dims='unc_Norm', attrs={'units': 'W yr m-2 TgN-1'})
    Par['ph_ari_VOC'] = xr.DataArray([-1.75E-5, 2.68E-5 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W yr m-2 Tg-1'})
    Par['ph_ari_CH4'] = xr.DataArray([-2.56E-6, 1.65E-6 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_ari_N2O'] = xr.DataArray([-3.70E-5, 2.78E-5 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_ari_EESC'] = xr.DataArray([-8.26E-9, 1.57E-9 / Cst.s1_to_p90], dims='unc_Norm', attrs={'units': 'W m-2 ppt-1'})

    ## natural emissions
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 6, 7, 8, 9 & 11)
    ## note: these are all attributed to ARI, but likely some of it is ACI
    ## note: dust and salt as total RF because emissions' orders of magnitude, and scaled to AOD
    ## note: preindustrial emissions obtained through pers. comm. from Bill Collins
    Par.coords['mod_ERF_dust'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_ERF_salt'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_ERF_DMS'] = ['UKESM1', 'NorESM2', 'GISS-E2-1']
    Par.coords['mod_ERF_BVOC'] = ['UKESM1', 'NorESM2', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par.coords['mod_ERF_LNOx'] = ['UKESM1', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Edust_pi = xr.DataArray([2749.3, 7875.92, 1104.82, 1646.589, 1975.828, 1765.5], dims='mod_ERF_dust', attrs={'units': 'Tg yr-1'})
    Esalt_pi = xr.DataArray([64756.92, 5504.602, 3586.251, 3711.25, 5654.69, 2606.], dims='mod_ERF_salt', attrs={'units': 'Tg yr-1'})
    Par['Ph_ari_dust'] = xr.DataArray(4.0/2.6 * np.array([3.1E-5, 3.8E-6, -1.7E-4, -1.1E-4, -0.2E-5, -8.2E-5]) * Edust_pi.values, dims='mod_ERF_dust', attrs={'units': 'W m-2'})
    Par['Ph_ari_salt'] = xr.DataArray(4.9/2.7 * np.array([-1.61E-4, -2.30E-4, -9.72E-5, -6.0E-4, -3.20E-4, -5.00E-4]) * Esalt_pi.values, dims='mod_ERF_salt', attrs={'units': 'W m-2'})
    Par['ph_ari_DMS'] = xr.DataArray([-0.0728, -0.0674, -0.0219], dims='mod_ERF_DMS', attrs={'units': 'W yr m-2 TgS-1'})
    Par['ph_ari_BVOC'] = xr.DataArray([0.4E-4, -11.8E-4, -9.7E-4, -5.4E-4, -1.3E-4], dims='mod_ERF_BVOC', attrs={'units': 'W yr m-2 Tg-1'})
    Par['ph_ari_LNOx'] = xr.DataArray([0.018-0.031, 0.036-0.034, 0.051-0.048, 0.021-0.023], dims='mod_ERF_LNOx', attrs={'units': 'W yr m-2 TgN-1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['Ph_ari_dust'] = xr.DataArray([-0.103, 0.166], dims='unc_Norm', attrs={'units': 'W m-2'})
        Par['Ph_ari_salt'] = xr.DataArray([-5.26, 6.20], dims='unc_Norm', attrs={'units': 'W m-2'})
        Par['ph_ari_DMS'] = xr.DataArray([-0.054, 0.023], dims='unc_LogNorm', attrs={'units': 'W yr m-2 TgS-1'})
        Par['ph_ari_BVOC'] = xr.DataArray([-5.6E-4, 4.7E-4], dims='unc_Norm', attrs={'units': 'W yr m-2 Tg-1'})
        Par['ph_ari_LNOx'] = xr.DataArray([-0.0025, 0.0063], dims='unc_Norm', attrs={'units': 'W yr m-2 TgN-1'})
        Par = Par.drop_dims(['mod_ERF_dust', 'mod_ERF_salt', 'mod_ERF_DMS', 'mod_ERF_BVOC', 'mod_ERF_LNOx'])

    ## aerosol lifetime sensivities to climate change
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 5, 6, 7 & 8)
    ## note: must convert the paper's parameters from finite difference to the exponential formula
    Par.coords['mod_tau_dust'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_tau_salt'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_tau_SO4'] = ['UKESM1', 'NorESM2', 'GISS-E2-1']
    Par['g_tau_dust'] = xr.DataArray(np.log(1 + np.array([2.6E-2, -0.4E-2, 1.9E-2, 1.0E-2, 3.7E-2, 1.6E-2]) * np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81])) / 
         np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81]), dims='mod_tau_dust', attrs={'units': 'K-1'})
    Par['g_tau_salt'] = xr.DataArray(np.log(1 + np.array([0.45E-2, -0.20E-2, -0.68E-2, -0.92E-2, 1.8E-2, -0.61E-2]) *  np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81])) / 
         np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81]), dims='mod_tau_salt', attrs={'units': 'K-1'})
    Par['g_tau_SO4'] = xr.DataArray(np.log(1 + np.array([2.48E-2, 2.73E-2, 1.13E-2]) * np.array([7.46, 3.96, 3.81])) / 
        np.array([7.46, 3.96, 3.81]), dims='mod_tau_SO4', attrs={'units': 'K-1'})
    ## turn into distribution (manually for nice rounding)
    ## note: keeping the Norm distrib for SO4 to span (slightly) more possibilities
    if nice_rounding:
        Par['g_tau_dust'] = xr.DataArray([0.016, 0.012], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_tau_salt'] = xr.DataArray([-0.000, 0.001], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_tau_SO4'] = xr.DataArray([0.020, 0.006] , dims='unc_Norm', attrs={'units': 'K-1'})
        Par = Par.drop_dims(['mod_tau_dust', 'mod_tau_salt', 'mod_tau_SO4'])


    ## aerosol-cloud interactions parameters
    ## (Smith et al., 2024; https://doi.org/10.5194/gmd-17-8569-2024) (Table 2)
    ## note: not using the KDE draw for now
    Par.coords['mod_ERF_aci'] = ['CanESM5', 'CNRM-CM6-1', 'E3SM-2-0', 'GFDL-CM4', 'GFDL-ESM4', 'GISS-E2-1-G', 'HadGEM3-GC31-LL', 'IPSL-CM6A-LR', 'MIROC6', 'MPI-ESM-1-2-HAM', 'MRI-ESM2-0', 'NorESM2-LM', 'UKESM1-0-LL']
    Par['Ph_aci'] = xr.DataArray([-0.856, -1.50, -1.44, -4507, -13202, -0.585, -0.941, -1.26, -1.03, -2.35, -7.74, -12527, -0.723], dims='mod_ERF_aci', attrs={'units': 'W m2'})
    Par['i_aci_SO2'] = xr.DataArray(np.array([0.0199, 0.00601, 0.0715, 1.10E-6, 2.54E-7, 0.00819, 0.0222, 0.00266, 0.0073, 0.00718, 0.000776, 6.91E-7, 0.0335]) * (Cst.m_SO2/Cst.m_S).values, dims='mod_ERF_aci', attrs={'units': 'yr TgS-1'})
    Par['i_aci_BC'] = xr.DataArray([0.394, 0.046, 1.29E-41, 5.94E-7, 2.70E-6, 1.28, 4.81E-33, 1.76E-16, 0.149, 3.85E-13, 0.00412, 2.78E-114, 8.76E-37], dims='mod_ERF_aci', attrs={'units': 'yr TgC-1'})
    Par['i_aci_OC'] = xr.DataArray([1.25E-16, 0.0111, 0.352, 2.13E-6, 6.07E-7, 5.36E-11, 0.0367, 0.0019, 6.27E-18, 0.00975, 5.27E-27, 1.62E-6, 6.38E-13], dims='mod_ERF_aci', attrs={'units': 'yr TgC-1'})


    ## radiative forcing from H2 (through O3, SWV, aerosols)
    ## (Sand et al., 2023; https://doi.org/10.1038/s43247-023-00857-8) (Supplementary Table 4)
    ## note: assumed to SARF with efficacy of 1 and/or ERF
    Par.coords['mod_ERF_H2'] = ['GFDL', 'INCA', 'OsloCTM', 'UKCA', 'WACCM']
    Par['ph_O3_H2'] = xr.DataArray([0.21E-3/6.98, 0.15E-3/7.36, 0.22E-3/6.23, 0.17E-3/7.80, 0.18E-3/5.36], dims='mod_ERF_H2', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_swv_H2'] = xr.DataArray([0.16E-3/6.98, 0.11E-3/7.36, 0.17E-3/6.23, np.nan, 0.05E-3/5.36], dims='mod_ERF_H2', attrs={'units': 'W m-2 ppb-1'})
    Par['ph_ari_H2'] = xr.DataArray([0.00E-3/6.98, 0.03E-3/7.36, -0.03E-3/6.23, np.nan, np.nan], dims='mod_ERF_H2', attrs={'units': 'W m-2 ppb-1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ph_O3_H2'] = xr.DataArray([2.82E-5, 0.61E-5], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
        Par['ph_swv_H2'] = xr.DataArray([1.86E-5, 0.70E-5], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
        Par['ph_ari_H2'] = xr.DataArray([-0.24E-6, 3.63E-6], dims='unc_Norm', attrs={'units': 'W m-2 ppb-1'})
        Par = Par.drop_dims('mod_ERF_H2')


    ## efficacy for volcano forcing
    ## (Gregory et al., 2016; https://doi.org/10.1007/s00382-016-3055-1) (Abstract)
    ## note: this is firstly to reduce the noise in the historical period
    Par['e_volc'] = xr.DataArray(0.6, attrs={'units':'1'})


    ## RETURN
    return Par

