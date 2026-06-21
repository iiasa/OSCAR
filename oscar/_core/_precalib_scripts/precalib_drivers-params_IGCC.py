import warnings
import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_precalib import run_precalib

from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_CH4 import get_params as get_params_CH4
from oscar._core._params_spec.Par_N2O import get_params as get_params_N2O
from oscar._core._params_spec.Par_strato import get_params as get_params_strato


name = 'drivers-params_IGCC'


## get time-lag parameter from strato chem.
t_lag = get_params_strato().t_lag

## get OH and hv present-day periods
years_OH = get_params_CH4().v_CH4_OH.years
years_hv = get_params_N2O().v_N2O_hv.years


##################################################
##   PRECALIBRATION OF DRIVERS-DERIVED PARAMS
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

    ## load reference datasets
    ds_conc = load_data('concentrations_IGCC')
    ds_temp = load_data('temperature_IGCC')

    ## initialization
    Par = xr.Dataset()


    ## present-day concentrations
    ## CH4, N2O, halogens
    years_dict = {'CH4': years_OH, 'N2O': years_hv, 'Xhalo': years_hv}
    for spc in ['CH4', 'N2O', 'Xhalo']:
        Par[f'{spc}_pd'] = ds_conc[spc].sel(year=slice(*years_dict[spc])).mean('year')
        Par[f'{spc}_pd'].attrs['units'] = ds_conc[spc].units
        Par[f'{spc}_pd'].attrs['years'] = years_dict[spc]
    ## lagged halogens
    Par['Xhalo_lag_pd'] = np.nan + xr.zeros_like(Par.spc_halo).astype(float) + xr.zeros_like(t_lag)
    for age_air in t_lag.age_air.values:
        if float.is_integer(float(t_lag.loc[age_air])):
            Par['Xhalo_lag_pd'].loc[:, age_air] = ds_conc.Xhalo.sel(year=slice(*[yr - int(t_lag.loc[age_air]) for yr in years_dict['Xhalo']])).mean('year')
        elif float.is_integer(float(t_lag.loc[age_air] - 0.5)): 
            Par['Xhalo_lag_pd'].loc[:, age_air] = ds_conc.Xhalo.rolling(year=2).mean().sel(year=slice(*[yr - int(t_lag.loc[age_air]) for yr in years_dict['Xhalo']])).mean('year')
    Par['Xhalo_lag_pd'].attrs['units'] = ds_conc[spc].units
    Par['Xhalo_lag_pd'].attrs['years'] = years_dict[spc]


    ## preindustrial and present-day land cover
    years_temp = (int(np.min(np.append(years_OH, years_hv))), int(np.max(np.append(years_OH, years_hv))))
    Par['D_Tg_pd'] = ds_temp.GSAT.sel(year=slice(*years_temp)).mean() - ds_temp.GSAT.sel(year=slice(1850, 1900)).mean()
    Par['D_Tg_pd'].attrs['units'] = ds_temp.GSAT.units
    Par['D_Tg_pd'].attrs['years'] = years_temp
    Par['D_Tg_pd'].attrs['ref_years'] = tuple([int(val) for val in ds_temp.GSAT.ref_years.split('-')])


    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params)

