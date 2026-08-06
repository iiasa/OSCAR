import warnings
import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions
from oscar._core._base.fct_drivers import format_LUH, format_CMIP_emis

from oscar._core._params_spec.Cst import Cst
from oscar._core._params_spec.Par_CH4 import get_params as get_params_CH4
from oscar._core._params_spec.Par_N2O import get_params as get_params_N2O
from oscar._core._params_spec.Par_strato import get_params as get_params_strato


name = 'drivers-params_CMIP7'


## get time-lag parameter from strato chem.
t_lag = get_params_strato().t_lag

## get OH and hv present-day periods
years_OH = get_params_CH4().v_CH4_OH_pd.years
years_hv = get_params_N2O().v_N2O_hv_pd.years

## definition of LUC present-day period
year_pd = 2022

## definition of preindustrial period
year_pi = 1750
years_pi = (1750, 1760) # because of IAV


##################################################
##   PRECALIBRATION OF DRIVERS-DERIVED PARAMS
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, keep_all_Ebb=False, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## load reference datasets
        ds_emis = aggreg_regions(load_data('emissions_CMIP7').sel(scen='hist', drop=True), mod_region)
        ds_land = aggreg_regions(load_data('land-use_LUH3').sel(scen='hist', drop=True), mod_region)
        ds_conc = load_data('concentrations_CMIP7').sel(scen='hist', drop=True)

        ## initialization
        Par = xr.Dataset()

        ## emissions
        ## get formatted params
        ds_tmp = format_CMIP_emis(ds_emis, 
            year_pi=year_pi, years_pi=years_pi, years_pd=years_OH, 
            scen_hist='hist', sect_endo=['bor', 'def', 'for', 'tem', 'gra'])
        ds_tmp = ds_tmp.drop_vars([var for var in ds_tmp if var.startswith('D_') and not var.endswith('_pd')])
        ## drop full Ebb
        if not keep_all_Ebb:
            ds_tmp = ds_tmp.drop_vars([var for var in ds_tmp if 'Ebb_' in var and var.endswith('_pi')])
            ds_tmp = ds_tmp.drop_vars([var for var in ds_tmp if 'Ebb2_' in var and var.endswith('_pd')])
        ## assign
        for var in ds_tmp:
            Par[var] = ds_tmp[var]


        ## land use
        ## get formatted params
        ds_tmp = format_LUH(ds_land, 
            year_pi=year_pi, year_pd=year_pd, 
            scen_hist='hist', keep_harv_dC=False)
        ds_tmp = ds_tmp.drop_vars([var for var in ds_tmp if var.startswith('D_')])
        ## assign
        for var in ds_tmp:
            Par[var] = ds_tmp[var]
        
        
        ## concentrations
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


    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

