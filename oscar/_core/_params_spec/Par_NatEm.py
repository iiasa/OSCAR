import numpy as np
import xarray as xr

from scipy.optimize import fsolve

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## natural emissions baselines and sensitivities
## based on AR6 WG1 and CMIP6/AerchemMIP
def get_params(nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogNorm'] = ['mean', 'std']
    Par.coords['unc_2HalfNorm'] = ['mean', 'std_neg', 'std_pos']


    ## preindustrial emissions
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021)
    ## note: obtained through pers. comm. from Bill Collins
    ## note: dust and salt should have same mod_ as ERF parameters, because of emissions' order of magnitude
    Par.coords['mod_ERF_dust'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_ERF_salt'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par['Edust_pi'] = xr.DataArray([2749.3, 7875.92, 1104.82, 1646.589, 1975.828, 1765.5], dims='mod_ERF_dust', attrs={'units': 'Tg yr-1'})
    Par['Esalt_pi'] = xr.DataArray([64756.92, 5504.602, 3586.251, 3711.25, 5654.69, 2606.], dims='mod_ERF_salt', attrs={'units': 'Tg yr-1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['Edust_pi'] = xr.DataArray([2850., 2300.], dims='unc_LogNorm', attrs={'units': 'Tg yr-1'})
        Par['Esalt_pi'] = xr.DataArray([14300., 22600.], dims='unc_LogNorm', attrs={'units': 'Tg yr-1'})
        Par = Par.drop_dims(['mod_ERF_dust', 'mod_ERF_salt'])

    ## (Griffiths et al., 2021; https://doi.org/10.5194/acp-21-4187-2021) (Section 2.3)
    ## note: very close to Thornhill et al., just one more model, but cited as ref in AR6
    Par['Enat_LNOx_pi'] = xr.DataArray([4.9, 1.9], dims='unc_LogNorm', attrs={'unit': 'TgN yr-1'})


    ## present-day emissions
    ## (Szopa et al., 2021; https://doi.org/10.1017/9781009157896.008) (Section 6.2.2)
    ## note: for BVOC, includes only isoprene and monoterpenes
    Par['Enat_DMS_pd'] = xr.DataArray(np.array([0.5*(18+24), 0.5*(24-18) / Cst.s1_to_p90]), dims='unc_LogNorm', attrs={'unit': 'TgS yr-1'})
    Par['Enat_BVOC_pd'] = xr.DataArray(np.array([0.5*((300+30)+(600+150)), 0.5*((600+150)-(300+30)) / Cst.s1_to_p95]) * (Cst.m_isop/5/Cst.m_C).values, dims='unc_LogNorm', attrs={'unit': 'Tg yr-1'})
    Par['Enat_SNOx_pd'] = xr.DataArray([0.5*(4.7+16.8), 0.5*(16.8-4.7) / Cst.s1_to_p95], dims='unc_LogNorm', attrs={'unit': 'TgN yr-1'})


    ## emissions (relative) sensitivities to climate change
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 6, 7, 8, 9 & 11)
    ## note: preindustrial emissions obtained through pers. comm. from Bill Collins
    Par.coords['mod_g_dust'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_g_salt'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_g_DMS'] = ['UKESM1', 'NorESM2', 'GISS-E2-1']
    Par.coords['mod_g_BVOC'] = ['UKESM1', 'NorESM2', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par.coords['mod_g_LNOx'] = ['UKESM1', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par['g_dust'] = xr.DataArray(np.array([65, -109, 70, -6., 181, 64]) / np.array([2749.3, 7875.92, 1104.82, 1646.589, 1975.828, 1765.5]), 
        dims='mod_g_dust', attrs={'units': 'K-1'})
    Par['g_salt'] = xr.DataArray(np.array([2570, 6.0, -3.93, 72, 258, -8.5]) / np.array([64756.92, 5504.602, 3586.251, 3711.25, 5654.69, 2606.]), 
        dims='mod_g_salt', attrs={'units': 'K-1'})
    Par['g_DMS'] = xr.DataArray(np.array([-0.04, -0.186, 0.02]) / np.array([32.443, 36.415, 54.109]) * (Cst.m_DMS/Cst.m_S).values, 
        dims='mod_g_DMS', attrs={'units': 'K-1'})
    Par['g_BVOC'] = xr.DataArray(np.array([32., 234, 81., 156, 113]) / np.array([757.353, 612.667, 483.505, 679.01, 1193.06]), 
        dims='mod_g_BVOC', attrs={'units': 'K-1'})
    Par['g_LNOx'] = xr.DataArray(np.array([0.27, -0.029, 0.336, 0.614]) / np.array([6.44, 3.44, 2.91, 7.20]), 
        dims='mod_g_LNOx', attrs={'units': 'K-1'})
    ## turn into distribution (manually for nice rounding)
    ## note: using HalfNorm distrib for LNOx, to cover the negative range that models don't
    if nice_rounding:
        Par['g_dust'] = xr.DataArray([0.033, 0.036], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_salt'] = xr.DataArray([0.017, 0.020], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_DMS'] = xr.DataArray([-0.0039, 0.0045], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_BVOC'] = xr.DataArray([0.18, 0.12], dims='unc_LogNorm', attrs={'units': 'K-1'})
        Par['g_LNOx'] = xr.DataArray([0.059, 2*0.047, 0.047], dims='unc_2HalfNorm', attrs={'units': 'K-1'})
        Par = Par.drop_dims(['mod_g_dust', 'mod_g_salt', 'mod_g_DMS', 'mod_g_BVOC', 'mod_g_LNOx'])


    ## emissions (relative) sensitivities to climate change
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Tables 6, 7, 8, 9 & 11)
    ## note: preindustrial emissions obtained through pers. comm. from Bill Collins
    ## note: must convert the paper's parameters from finite difference to the exponential formula
    Par.coords['mod_g_dust'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_g_salt'] = ['CNRM-ESM2-1', 'UKESM1', 'MIROC6', 'NorESM2', 'GFDL-ESM4', 'GISS-E2-1']
    Par.coords['mod_g_DMS'] = ['UKESM1', 'NorESM2', 'GISS-E2-1']
    Par.coords['mod_g_BVOC'] = ['UKESM1', 'NorESM2', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par.coords['mod_g_LNOx'] = ['UKESM1', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par['g_dust'] = xr.DataArray(np.log(1 + np.array([65, -109, 70, -6., 181, 64]) / np.array([2749.3, 7875.92, 1104.82, 1646.589, 1975.828, 1765.5]) * 
        np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81])) / np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81]), dims='mod_g_dust', attrs={'units': 'K-1'})
    Par['g_salt'] = xr.DataArray(np.log(1 + np.array([2570, 6.0, -3.93, 72, 258, -8.5]) / np.array([64756.92, 5504.602, 3586.251, 3711.25, 5654.69, 2606.]) * 
        np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81])) / np.array([6.09, 7.46, 4.01, 3.96, 3.93, 3.81]), dims='mod_g_salt', attrs={'units': 'K-1'})
    Par['g_DMS'] = xr.DataArray(np.log(1 + np.array([-0.04, -0.186, 0.02]) / np.array([32.443, 36.415, 54.109]) * (Cst.m_DMS/Cst.m_S).values * 
        np.array([7.46, 3.96, 3.81])) / np.array([7.46, 3.96, 3.81]), dims='mod_g_DMS', attrs={'units': 'K-1'})
    Par['g_BVOC'] = xr.DataArray(np.log(1 + np.array([32., 234, 81., 156, 113]) / np.array([757.353, 612.667, 483.505, 679.01, 1193.06]) * 
        np.array([7.46, 3.96, 3.93, 6.49, 3.81])) / np.array([7.46, 3.96, 3.93, 6.49, 3.81]), dims='mod_g_BVOC', attrs={'units': 'K-1'})
    Par['g_LNOx'] = xr.DataArray(np.log(1 + np.array([0.27, -0.029, 0.336, 0.614]) / np.array([6.44, 3.44, 2.91, 7.20]) * 
        np.array([7.46, 3.93, 6.49, 3.81])) / np.array([7.46, 3.93, 6.49, 3.81]), dims='mod_g_LNOx', attrs={'units': 'K-1'})
    ## turn into distribution (manually for nice rounding)
    ## note: using HalfNorm distrib for LNOx, to cover the negative range that models don't
    if nice_rounding:
        Par['g_dust'] = xr.DataArray([0.029, 0.032], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_salt'] = xr.DataArray([0.015, 0.018], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_DMS'] = xr.DataArray([-0.0039, 0.0045], dims='unc_Norm', attrs={'units': 'K-1'})
        Par['g_BVOC'] = xr.DataArray([0.12, 0.07], dims='unc_LogNorm', attrs={'units': 'K-1'})
        Par['g_LNOx'] = xr.DataArray([0.047, 2*0.037, 0.037], dims='unc_2HalfNorm', attrs={'units': 'K-1'})
        Par = Par.drop_dims(['mod_g_dust', 'mod_g_salt', 'mod_g_DMS', 'mod_g_BVOC', 'mod_g_LNOx'])


    ## sensitivity of BVOC to global forest area loss
    ## (Szopa et al., 2021; https://doi.org/10.1017/9781009157896.008) (Section 6.2.2.3)
    ## note: change in global forest area between 1850 and 2004 extracted from LUH1 and rounded
    ## note: limiting uncertainty range to prevent parameter > 1
    f_Afor_pd = 1 - 3960 / 4850.
    Par['i_BVOC_Afor'] = xr.DataArray([0.175/f_Afor_pd, 1 - 0.175/f_Afor_pd], dims='unc_LogitNorm', attrs={'units': '1'})


    ## RETURN
    return Par

