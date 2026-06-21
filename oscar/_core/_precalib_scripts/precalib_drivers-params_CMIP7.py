import warnings
import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_CH4 import get_params as get_params_CH4
from oscar._core._params_spec.Par_N2O import get_params as get_params_N2O
from oscar._core._params_spec.Par_strato import get_params as get_params_strato


name = 'drivers-params_CMIP7'


## get time-lag parameter from strato chem.
t_lag = get_params_strato().t_lag

## get OH and hv present-day periods
years_OH = get_params_CH4().v_CH4_OH.years
years_hv = get_params_N2O().v_N2O_hv.years

## definition of LUC present-day period
year_PD = 2023

## definition of preindustrial period
year_PI = 1750
years_PI = (1750, 1760) # because of IAV


##################################################
##   PRECALIBRATION OF DRIVERS-DERIVED PARAMS
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

    ## load reference datasets
    ds_emis = load_data('emissions_CMIP7').sum('reg_code', min_count=1, keep_attrs=True).sel(scen='hist', drop=True)
    ds_conc = load_data('concentrations_CMIP7').sel(scen='hist', drop=True)
    ds_land = aggreg_regions(load_data('land-use_LUH3').sel(scen='hist', drop=True), mod_region)

    ## initialization
    Par = xr.Dataset()


    ## preindustrial emissions of precursors
    ## anthropogenic
    for spc in ['SO2', 'BC', 'OC', 'NOx', 'CO', 'VOC']:
        Par[f'Eant_{spc}_pi'] = ds_emis[f'Eant_{spc}'].sel(year=year_PI, drop=True).sum('sect', min_count=1)
        Par[f'Eant_{spc}_pi'].attrs['units'] = ds_emis[f'Eant_{spc}'].units
        Par[f'Eant_{spc}_pi'].attrs['year'] = year_PI
    ## aviation and shipping
    for sec in ['air', 'shp']:
        Par[f'E{sec}_NOx_pi'] = ds_emis['Eant_NOx'].sel(year=year_PI, sect=sec, drop=True)
        Par[f'E{sec}_NOx_pi'].attrs['units'] = ds_emis['Eant_NOx'].units
        Par[f'E{sec}_NOx_pi'].attrs['year'] = year_PI
    ## biomass burning
    for spc in ['SO2', 'BC', 'OC', 'NOx', 'CO', 'VOC']:
        Par[f'Ebb_{spc}_pi'] = ds_emis[f'Eant_{spc}'].sel(year=slice(*years_PI), drop=True).sum('sect', min_count=1).mean('year')
        Par[f'Ebb_{spc}_pi'].attrs['units'] = ds_emis[f'Ebb_{spc}'].units
        Par[f'Ebb_{spc}_pi'].attrs['years'] = years_PI


    ## present-day emissions of precursors (delta)
    ## anthropogenic
    for spc in ['NOx', 'CO', 'VOC']:
        Par[f'D_Eant_{spc}_pd'] = ds_emis[f'Eant_{spc}'].sel(year=slice(*years_OH), drop=True).sum('sect', min_count=1).mean('year')
        Par[f'D_Eant_{spc}_pd'] -= Par[f'Eant_{spc}_pi']
        Par[f'D_Eant_{spc}_pd'].attrs['units'] = ds_emis[f'Eant_{spc}'].units
        Par[f'D_Eant_{spc}_pd'].attrs['years'] = years_OH
    ## aviation and shipping
    for sec in ['air', 'shp']:
        Par[f'D_E{sec}_NOx_pd'] = ds_emis['Eant_NOx'].sel(year=slice(*years_OH), sect=sec, drop=True).mean('year')
        Par[f'D_E{sec}_NOx_pd'] -= Par[f'E{sec}_NOx_pi']
        Par[f'D_E{sec}_NOx_pd'].attrs['units'] = ds_emis['Eant_NOx'].units
        Par[f'D_E{sec}_NOx_pd'].attrs['years'] = years_OH
    ## biomass burning
    for spc in ['NOx', 'CO', 'VOC']:
        Par[f'D_Ebb_{spc}_pd'] = ds_emis[f'Eant_{spc}'].sel(year=slice(*years_OH), drop=True).sum('sect', min_count=1).mean('year')
        Par[f'D_Ebb_{spc}_pd'] -= Par[f'Ebb_{spc}_pi']
        Par[f'D_Ebb_{spc}_pd'].attrs['units'] = ds_emis[f'Ebb_{spc}'].units
        Par[f'D_Ebb_{spc}_pd'].attrs['years'] = years_OH


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
    Par['Aland_pi'] = (ds_land.Aland1 + ds_land.Aland2).sel(year=year_PI, drop=True)
    Par['Aland_pi'].attrs['year'] = year_PI
    Par['Aland_pd'] = (ds_land.Aland1 + ds_land.Aland2).sel(year=year_PD, drop=True)
    Par['Aland_pd'].attrs['year'] = year_PD

    ## preindustrial land use activities
    Par['dA_harv_pi'] = (ds_land.dA_harv1 + ds_land.dA_harv2).sel(year=year_PI, drop=True)
    Par['dA_harv_pi'].attrs['year'] = year_PI
    Par['dA_shift_pi'] = ds_land.dA_shift.sel(year=year_PI, drop=True)
    Par['dA_shift_pi'].attrs['year'] = year_PI


    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

