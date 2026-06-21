import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## permafrost carbon parameters
## (Gasser et al., 2018; https://doi.org/10.1038/s41561-018-0227-0) (Table S4)
## additional calibration on UVic (MacDougall, 2021; https://doi.org/10.5194/bg-18-4937-2021)
## climate uncertainty from IMOGEN (Burke et al., 2017; https://doi.org/10.5194/bg-14-3051-2017)
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()
    Par.coords['unc_LogitNorm'] = ['mean', 'std']
    Par.coords['mod_Epf'] = ['JSBACH', 'ORCHIDEE-MICT', 'JULES-DeepResp', 'JULES-SuppressResp', 'UVic']
    Par.coords['mod_Epf_clim'] = ['BCCR-BCM2-0', 'CCCma_CGCM3-1', 'CNRM-CM3', 'CSIRO-Mk3-0', 'CSIRO-Mk3-5', 'GFDL-CM2-0', 'GFDL-CM2-1', 'GISS-E-H', 'GISS-E-R', 'IAP-FGOALS1-0-g', 'INGV-ECHAM4', 'IPSL-CM4', 'MIROC3-2-hires', 'MIROC3-2-medres', 'MIUB-ECHO-G', 'MPI-ECHAM5', 'MRI-CGCM2-3-2a', 'NCAR-CCSM3-0', 'NCAR-PCM1', 'UKMO-HadCM3', 'UKMO-HadGEM1']
    Par.coords['reg_pf'] = ['Eurasia', 'North America']
    Par.coords['box_thaw'] = np.arange(3)


    ## polar amplification of local air temperature
    ## with additional uncertainty from range of IMOGEN emulation
    Par['a_Tpf_Tg'] = xr.DataArray([[1.86, 1.95], [1.87, 1.78], [1.96, 2.03], [1.96, 2.02], [1.61, 1.76]], 
        dims=['mod_Epf', 'reg_pf'])
    Par['a_Tpf_Tg'] = Par.a_Tpf_Tg * xr.DataArray([
        [0.913, 0.864], [0.884, 0.902], [0.899, 0.836], [0.889, 1.082], [0.873, 0.961], [0.935, 0.884], [1.002, 0.988],
        [0.825, 0.823], [0.849, 0.918], [1.042, 1.135], [1.208, 1.202], [0.913, 1.121], [0.862, 0.982], [1.064, 1.021], 
        [1.356, 0.966], [1.069, 0.934], [0.871, 0.964], [1.170, 1.192], [1.125, 1.097], [1.023, 0.932], [1.229, 1.197]], 
        dims=['mod_Epf_clim', 'reg_pf'])
    Par['a_Tpf_Tg'].attrs['units'] = '1'
    

    ## preindustrial frozen permafrost carbon pool
    Par['Cfroz_pi'] = xr.DataArray([[414., 277.], [271., 118.], [481., 176.], [373., 121.], [371., 129.]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'PgC'})

    ## permafrost respiration sensitivities to temperature
    Par['g_ethaw_T'] = xr.DataArray([[0.0969, 0.101], [0.114, 0.110], [0.168, 0.144], [0.135, 0.112], [0.134, 0.139]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'K-1'})
    Par['g_ethaw_T2'] = xr.DataArray([[0.00206, 0.00214], [0.00366, 0.00345], [0.00531, 0.00380], [0.00499, 0.00339], [0.00155, 0.00149]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'K-2'})

    ## permafrost respiration buffer factor
    Par['x_ethaw'] = xr.DataArray([[0.938, 0.735], [3.38, 3.71], [1.37, 1.58], [0.662, 2.19], [0.524, 0.464]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': '1'})

    ## minimum (negative) thawed fraction
    Par['pthaw_min'] = xr.DataArray([[0.209, 0.195], [0.0807, 0.875], [0.772, 0.959], [0.870, 1.16], [0.0139, 0.0158]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': '1'})

    ## thawed fraction sensitivity to temperature
    Par['g_pthaw'] = xr.DataArray([[0.176, 0.286], [1.62, 0.0973], [0.153, 0.145], [0.180, 0.190], [3190., 6680.]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'K-1'})

    ## shape parameter for thawed fraction function
    Par['x_pthaw'] = xr.DataArray([[2.30, 1.42], [0.357, 8.44], [2.05, 1.91], [1.63, 1.68], [2.00E-4, 8.56E-5]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': '1'})

    ## speed of thawing and freezing
    Par['v_thaw'] = xr.DataArray([[4.89, 0.500], [0.287, 0.500], [0.266, 0.599], [0.373, 0.492], [0.140, 0.129]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'yr-1'})
    Par['v_froz'] = xr.DataArray([[0., 0.], [0.0529, 0.0490], [0.0410, 0.0330], [0.0613, 0.0429], [0.00151, 0.00378]], 
        dims=['mod_Epf', 'reg_pf'], attrs={'units': 'yr-1'})
 
    ## partitioning coefficient of thawed carbon
    Par['p_fthaw'] = xr.DataArray(
        [[[0.070, 0.100], [0.244, 0.288], [0.034, 0.651], [1., 1.], [0., 0.051]],
        [[0.264, 0.234], [0.756, 0.712], [0.966, 0.349], [0., 0.], [0.125, 0.068]],
        [[0.666, 0.666], [0., 0.], [0., 0.], [0., 0.], [0.875, 0.881]]],
        dims=['box_thaw', 'mod_Epf', 'reg_pf'], attrs={'units': '1', 'mod_noise_override': 0.})

    ## time-scales of emission of thawed carbon
    Par['t_ethaw'] = xr.DataArray(
        [[[7.96, 11.6], [1330., 2950.], [156., 1970.], [9430., 4320.], [1E18, 12.0]],
        [[88.5, 103.], [15000., 23400.], [3160., 5370.], [1E18, 1E18], [20.2, 23.6]],
        [[1360, 1240], [1E18, 1E18], [1E18, 1E18], [1E18, 1E18], [2410., 2370.]]],
        dims=['box_thaw', 'mod_Epf', 'reg_pf'], attrs={'units': 'yr'})

    ## fraction of instantaneous permafrost emissions
    Par['p_pf_inst'] =  xr.DataArray(0., attrs={'units': '1'}) 


    ## fraction of permafrost emitted as methane
    ## (Schuur et al., 2015; https://doi.org/10.1038/nature14338)
    ## note: arbitrary uncertainty
    Par['p_pf_CH4'] =  xr.DataArray([0.023, 0.023/2], dims='unc_LogitNorm',  attrs={'units': '1'})


    ## RETURN
    return Par

