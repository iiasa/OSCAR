import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'wetland_WETCHIMP'


##################################################
##   PRECALIBRATION OF WETLANDS
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data on selected regions
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ds = aggreg_regions(TMP, mod_region, weight_var={'Tl': '_area', 'Pl': '_area'})
            ds = ds.compute()

        ## SELECT:
        ## only models with both emissions and wetlands on all experiments
        keep = ds.Ewet.notnull().any('reg_land').all('exp') & ds.Awet.notnull().any('reg_land').all('exp')
        ds = ds.where(keep).dropna('model', how='all')

        ## ASSUMPTION:
        ## minimum wetland extent is 1 m2 (to help with log)
        ds['Awet'] = xr.full_like(ds['Awet'], 1E-10).where((ds['Awet'] > 0).any('exp') & (ds['Awet'] == 0), ds['Awet'])  

        ## initialization
        Par = xr.Dataset()

        ## baseline conditions
        Par['CO2_piW'] = ds.CO2.sel(exp='exp1', drop=True)
        Par['Tl_piW'] = ds.Tl.sel(exp='exp1', drop=True)
        Par['Pl_piW'] = ds.Pl.sel(exp='exp1', drop=True)
        Par['Awet_piW'] = ds.Awet.sel(exp='exp1', drop=True)
        Par['ewet_piW'] = ds.Ewet.sel(exp='exp1', drop=True) / ds.Awet.sel(exp='exp1', drop=True)
        Par['ewet_piW'].attrs['units'] = 'TgC Mha-1 yr-1'

        ## wetland areal emissions parameters
        Par['b_ewet_CO2'] = (ds.Ewet.sel(exp='exp4') / ds.Ewet.sel(exp='exp1') * ds.Awet.sel(exp='exp1') / ds.Awet.sel(exp='exp4') - 1) / np.log(ds.CO2.sel(exp='exp4') / ds.CO2.sel(exp='exp1'))
        Par['b_ewet_CO2'].attrs['units'] = '1'
        Par['g_ewet_T'] = np.log(ds.Ewet.sel(exp='exp5') / ds.Ewet.sel(exp='exp1') * ds.Awet.sel(exp='exp1') / ds.Awet.sel(exp='exp5')) / (ds.Tl.sel(exp='exp5') - ds.Tl.sel(exp='exp1'))
        Par['g_ewet_T'].attrs['units'] = 'K-1'
        Par['x_ewet_P'] = np.log(ds.Ewet.sel(exp='exp6') / ds.Ewet.sel(exp='exp1') * ds.Awet.sel(exp='exp1') / ds.Awet.sel(exp='exp6')) / np.log(ds.Pl.sel(exp='exp6') / ds.Pl.sel(exp='exp1'))
        Par['x_ewet_P'].attrs['units'] = '1'

        ## wetland extent parameters
        Par['x_Awet_CO2'] = np.log(ds.Awet.sel(exp='exp4') / ds.Awet.sel(exp='exp1')) / np.log(ds.CO2.sel(exp='exp4') / ds.CO2.sel(exp='exp1'))
        Par['x_Awet_CO2'].attrs['units'] = '1'
        Par['x_Awet_T'] = np.log(ds.Awet.sel(exp='exp5') / ds.Awet.sel(exp='exp1')) / np.log(ds.Tl.sel(exp='exp5') / ds.Tl.sel(exp='exp1'))
        Par['x_Awet_T'].attrs['units'] = '1'
        Par['x_Awet_P'] = np.log(ds.Awet.sel(exp='exp6') / ds.Awet.sel(exp='exp1')) / np.log(ds.Pl.sel(exp='exp6') / ds.Pl.sel(exp='exp1'))
        Par['x_Awet_P'].attrs['units'] = '1'

        ## ensure physically meaningful parameters
        ## no negative fertilisation
        Par['b_ewet_CO2'] = Par['b_ewet_CO2'].where(Par['b_ewet_CO2'] > 0, 0.)
        ## zero values (i.e. no wetlands) instead of nan if at least one model has values
        for var in Par:
            Par[var] = Par[var].fillna(0.).where(Par[var].notnull().any('model'), Par[var])
        ## zero sensitivies if ref area or emissions are zero
        for var in ['ewet_piW', 'b_ewet_CO2', 'g_ewet_T', 'x_ewet_P', 'x_Awet_CO2', 'x_Awet_T', 'x_Awet_P']:
            Par[var] = Par[var].where(Par['Awet_piW'] != 0, 0.)
            Par[var] = Par[var].where(Par['ewet_piW'] != 0, 0.)

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

        ## make dimensions
        Par = Par.rename({'model': 'mod_Ewet'})

    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

