import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


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


    ## OH lifetime sensitivities
    ## (Holmes et al., 2013; https://doi.org/10.5194/acp-13-285-2013) (Table 2)
    ## note: taking opposite values because sensivity of v = 1/tau
    Par['ch_OH_CH4'] = xr.DataArray([-0.31, 0.04], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_O3'] = xr.DataArray([-0.55, 0.11], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_Tt'] = xr.DataArray([3.0, 0.8], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_Qt'] = xr.DataArray([0.32, 0.03], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_BB'] = xr.DataArray([-0.020, 0.015], dims='unc_Norm', attrs={'units': '1'})
    Par['ch_OH_NOx_lit'] = xr.DataArray([0.16, 0.06], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_NOx'] = xr.DataArray([0.14, 0.03], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_NOx_shp'] = xr.DataArray([0.03, 0.015], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_NOx_air'] = xr.DataArray([0.014, 0.003], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_CO'] = xr.DataArray([-0.06, 0.02], dims='unc_LogNorm', attrs={'units': '1'})
    Par['ch_OH_VOC'] = xr.DataArray([-0.04, 0.01], dims='unc_LogNorm', attrs={'units': '1'})


    ## climate-related parameters
    ## (Holmes et al., 2013; https://doi.org/10.5194/acp-13-285-2013) (Supplementary Information)
    ## present-day tropospheric temperature
    Par['Tt_pd'] = xr.DataArray(251.9, attrs={'units': 'K', 'years': (2000, 2009)})
    ## scaling factors for tropospheric temperature and specific humidity changes
    Par['a_Tt'] = xr.DataArray([0.94, 0.1], dims='unc_LogNorm', attrs={'units':'1'})
    Par['a_Qt'] = xr.DataArray([1.5, 0.1], dims='unc_LogNorm', attrs={'units':'1'})


    ## sensitivity to H2
    ## (Sand et al., 2023; https://doi.org/10.1038/s43247-023-00857-8) (Supplementary Tables 2 and 4)
    ## note: likely typo for UKCA H2 flux in Table 4 (should be same as Table 3: 6.87 instead of 11.3)
    ## note: GFDL did +40% instead of +10%
    Par.coords['mod_ch_OH_H2'] = ['GFDL', 'INCA', 'OsloCTM', 'UKCA', 'WACCM']
    Par['ch_OH_H2'] = xr.DataArray(
        np.log1p([-30.5*0.26/528, -7.23*0.29/593, -8.54*0.31/683, -6.87*0.38/624, -9.93*0.33/727]) / 
        np.log([1.4, 1.1, 1.1, 1.1, 1.1]), 
        dims='mod_ch_OH_H2', attrs={'units': '1'})
    ## turn into distribution (manually for nice rounding)
    if nice_rounding:
        Par['ch_OH_H2'] = xr.DataArray([-0.043, 0.004], dims='unc_LogNorm', attrs={'units': '1'})
        Par = Par.drop_dims('mod_ch_OH_H2')


    ## preindustrial natural emissions
    ## (Szopa et al., 2021; https://doi.org/10.1017/9781009157896.008) (Section 6.2.2)
    ## note: not all of them used at this stage
    Par['Elit_NOx_pi'] = xr.DataArray([0.5*(3.2+7.6), 0.5*(7.6-3.2) / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'unit': 'TgN yr-1'})
    Par['Esoil_NOx_pi'] = xr.DataArray([0.5*(4.7+16.8), 0.5*(16.8-4.7) / Cst.s1_to_p90], dims='unc_LogNorm', attrs={'unit': 'TgN yr-1'})
    Par['Ebio_VOC_pi'] = xr.DataArray(np.array([0.5*((300+30)+(600+150)), 0.5*((600+150)-(300+30)) / Cst.s1_to_p90]) / 0.882, dims='unc_LogNorm', attrs={'unit': 'Tg yr-1'})
    Par['Eocean_DMS_pi'] = xr.DataArray(np.array([0.5*(18+24), 0.5*(24-18) / Cst.s1_to_p90]) * (Cst.m_S/Cst.m_DMS).values, dims='unc_LogNorm', attrs={'unit': 'Tg yr-1'})

    ## present-day natural emissions
    ## note: assumed
    Par['D_Elit_NOx_pd'] =  xr.DataArray(0., attrs={'unit': 'TgN yr-1'})


    ## RETURN
    return Par

