import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_halo import get_params as get_params_halo
from oscar._core._params_spec.Par_strato import get_params as get_params_strato


## get halogenated compounds parameters
Par_halo = get_params_halo()
Par_strato = get_params_strato()


##==================
##==================

## parameters for tropospheric chemistry
## compiled from various sources
def get_params(nice_rounding=True, **useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    Par.coords['unc_Norm'] = ['mean', 'std']
    Par.coords['unc_LogNorm'] = ['mean', 'std']


    ## OH lifetime sensitivities to anthropogenic factors
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-853-2021) (Table S9)
    ## note: assumes all lifetime changes attributable to OH sink
    ## note: perturbations data extracted from RCMIP phase 2 inputs
    Par.coords['mod_ch_OH_ant'] = ['UKESM1', 'CESM2-WACCM', 'GFDL-ESM4', 'BCC', 'GISS-E2', 'MRI-ESM2']
    Par['ch_OH_CH4'] = xr.DataArray(-np.log1p([22E-2, 22E-2, 21E-2, 26E-2, 18E-2, 22E-2]) / np.log(1831.47 / 808.25), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})
    Par['ch_OH_N2O'] = xr.DataArray(-np.log1p([-1.2E-2, -2.8E-2, np.nan, np.nan, -3.9E-2, -1.3E-2]) / np.log(326.99 / 273.02), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})
    Par['ch_OH_NOx'] = xr.DataArray(-np.log1p([-25E-2, -35E-2, -33E-2, np.nan, -46E-2, -26E-2]) / np.log(155.64 / 13.46), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})

    ## note: CO and VOC split following older parameterization:
    ## (Holmes et al., 2013; https://doi.org/10.5194/acp-13-285-2013) (Table 2)
    r_CO_VOC = 0.06 / 0.04
    Par['ch_OH_CO'] = xr.DataArray(-np.log1p([11E-2, np.nan, 15E-2, np.nan, 27E-2, 21E-2]) / (np.log(964.05 / 411.46) + 1/r_CO_VOC * np.log(233.53 / 66.78)), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})
    Par['ch_OH_VOC'] = xr.DataArray(-np.log1p([11E-2, np.nan, 15E-2, np.nan, 27E-2, 21E-2]) / (r_CO_VOC * np.log(964.05 / 411.46) + np.log(233.53 / 66.78)), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})
    
    ## note: extra info needed for halocarbons
    ## ODS list: CFC-11, CFC-12, CFC-113, CFC-114, CFC-115, HCFC-22, HCFC-141b, HCFC-142b, CH3CCl3, CCl4, CH3Cl, CH2Cl2, CHCl3, CH3Br, Halon-1211, Halon-1202, Halon-1301, Halon-2402
    ODS_1850 = [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.03, 457., 6.91, 6.00, 5.30, 0.00, np.nan, 0.00, 0.00]
    ODS_2014 = [233.08, 520.58, 72.71, 16.31, 8.43, 229.54, 23.81, 22.08, 3.68, 83.07, 539.54, 36.35, 9.90, 6.69, 3.75, np.nan, 3.30, 0.43]
    EESC_1850 = ((Par_strato.p_fracrel * (Par_halo.n_Cl + Par_strato.a_Cl_Br * Par_halo.n_Br)).sel(age_air='mid_lat').values * ODS_1850).sum(where=~np.isnan(ODS_1850) & ~np.isnan(Par_strato.p_fracrel[0].values))
    EESC_2014 = ((Par_strato.p_fracrel * (Par_halo.n_Cl + Par_strato.a_Cl_Br * Par_halo.n_Br)).sel(age_air='mid_lat').values * ODS_2014).sum(where=~np.isnan(ODS_2014) & ~np.isnan(Par_strato.p_fracrel[0].values))
    Par['ch_OH_EESC'] = xr.DataArray(-np.log1p([-4.9E-2, np.nan, -7.5E-2, np.nan, -0.6E-2, -2.4E-2]) / np.log(EESC_2014 / EESC_1850), 
        dims='mod_ch_OH_ant', attrs={'units': '1'})
    
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ch_OH_CH4'] = xr.DataArray([-0.24, 0.02], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_N2O'] = xr.DataArray([0.13, 0.06], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_NOx'] = xr.DataArray([0.17, 0.05], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_CO'] = xr.DataArray([-0.10, 0.03], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_VOC'] = xr.DataArray([-0.067, 0.020], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_EESC'] = xr.DataArray([0.028, 0.019], dims='unc_LogNorm', attrs={'units': '1'})
        Par = Par.drop_dims('mod_ch_OH_ant')


    ## OH lifetime sensitivity to climate change
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Table 14)
    ## note: preindustrial temperature extracted by myself
    Par.coords['mod_ch_OH_Tg'] = ['UKESM1', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par['ch_OH_Tg'] = xr.DataArray(-np.log1p(np.array([-4.51E-2, -4.63E-2, +3.98E-2, -1.67E-2]) * np.array([7.46, 3.93, 6.49, 3.81])) / 
        np.log1p(np.array([7.46, 3.93, 6.49, 3.81]) / np.array([286.5, 286.6, 287.1, 287.6])), 
        dims='mod_ch_OH_Tg', attrs={'units': '1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ch_OH_Tg'] = xr.DataArray([6.4, 10.5], dims='unc_Norm', attrs={'units': '1'})
        Par = Par.drop_dims('mod_ch_OH_Tg')


    ## OH lifetime sensitivites to natural emissions
    ## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021) (Table 12)
    ## note: preindustrial emissions obtained through pers. comm. from Bill Collins
    ## note: must convert the paper's parameters from finite difference to the power-law formula (under doubling of emissions)
    Par.coords['mod_ch_OH_nat'] = ['UKESM1', 'GFDL-ESM4', 'CESM2-WACCM', 'GISS-E2-1']
    Par['ch_OH_BVOC'] = xr.DataArray(-np.log1p(np.array([0.033E-2, 0.030E-2, 0.035E-2, 0.018E-2]) * np.array([757.353, 483.505, 679.01, 1193.06])) / np.log(2.), 
        dims='mod_ch_OH_nat', attrs={'units': '1'})
    Par['ch_OH_LNOX'] = xr.DataArray(-np.log1p(np.array([-2.4E-2, -3.8E-2, -6.8E-2, -6.1E-2]) * np.array([7.46, 3.93, 6.49, 3.81])) / np.log(2.), 
        dims='mod_ch_OH_nat', attrs={'units': '1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ch_OH_BVOC'] = xr.DataArray([-0.28, 0.05], dims='unc_LogNorm', attrs={'units': '1'})
        Par['ch_OH_LNOx'] = xr.DataArray([0.16, 0.05], dims='unc_LogNorm', attrs={'units': '1'})
        Par = Par.drop_dims('mod_ch_OH_nat')
    

    ## OH lifetime sensitivity to H2
    ## (Sand et al., 2023; https://doi.org/10.1038/s43247-023-00857-8) (Supplementary Tables 2 and 4)
    ## note: likely typo for UKCA H2 flux in Table 4 (should be same as Table 3: 6.87 instead of 11.3)
    ## note: GFDL did +40% instead of +10%
    Par.coords['mod_ch_OH_H2'] = ['GFDL', 'INCA', 'OsloCTM', 'UKCA', 'WACCM']
    Par['ch_OH_H2'] = xr.DataArray(np.log1p([-30.5*0.26/528, -7.23*0.29/593, -8.54*0.31/683, -6.87*0.38/624, -9.93*0.33/727]) / 
        np.log([1.4, 1.1, 1.1, 1.1, 1.1]), dims='mod_ch_OH_H2', attrs={'units': '1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ch_OH_H2'] = xr.DataArray([-0.043, 0.004], dims='unc_LogNorm', attrs={'units': '1'})
        Par = Par.drop_dims('mod_ch_OH_H2')


    ## RETURN
    return Par

