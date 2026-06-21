import sys
import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib, get_best_fit
from oscar._core._base.fct_regions import aggreg_regions
path_precalib_in = get_paths()["precalib_data"]

name = 'regional-climate_CMIP6'


##################################################
##   PRECALIBRATION OF REGIONAL CLIMATE
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        with xr.open_dataset(path_precalib_in + '{}.nc'.format(name.replace('regional-',''))) as TMP:
            ds = TMP.sel(exp=['piControl', 'historical'] + [exp for exp in TMP.exp.values if 'ssp' in exp])
            ds = ds.drop_vars([var for var in ds if var not in ['tas', 'pr', '_area']]).dropna('year', how='all')
            ds = aggreg_regions(ds, mod_region, weight_var={'tas': '_area', 'pr': '_area'})
            ds = ds.compute()

        ## global data
        ds = ds.rename({'tas': 'lst', 'pr': 'lsp'})
        ds['tas'] = (ds['lst'] * ds['_area']).sum('reg_land', min_count=1) / ds['_area'].sum('reg_land', min_count=1)
        ds['pr'] = (ds['lsp'] * ds['_area']).sum('reg_land', min_count=1) / ds['_area'].sum('reg_land', min_count=1)
        ds = ds.drop_vars('_area')

        ## offset with preindustrial
        ds_delta = ds.sel(exp=[exp for exp in ds.exp.values if exp != 'piControl']) - ds.sel(exp='piControl', drop=True)

        ## initialization
        Par = xr.Dataset()
        for var in ['a_Tl_Tg', 'a_Pl_Pg', 'a_Pl_Tl']:
            Par[var] = np.nan + xr.zeros_like(ds.reg_land, dtype=np.float32) + xr.zeros_like(ds.model, dtype=np.float32)


        ## STEP 0
        ## preindustrial climate
        Par['Tg_pi'] = ds['tas'].sel(exp='piControl', drop=True).mean('year')
        Par['Pg_pi'] = ds['pr'].sel(exp='piControl', drop=True).mean('year')
        Par['Tl_pi'] = ds['lst'].sel(exp='piControl', drop=True).mean('year')
        Par['Pl_pi'] = ds['lsp'].sel(exp='piControl', drop=True).mean('year')


        ## STEP 1
        ## temperature

        ## make pattern scaling model
        def f_lst(tas, a):
            return a * tas

        ## make parameters
        params0 = {'a': dict(value=0, default=0)}

        ## loop on models and regions
        for mod in ds_delta.model.values:
            for reg in ds_delta.reg_land.values:
                print('\n', 'lst', mod, reg, '\n')

                ## sub dataset and ignore empty regions
                ds_tmp = ds_delta.sel(reg_land=reg, model=mod).dropna('year', how='all')
                if all([ds_tmp[var].isnull().sum() < ds_tmp[var].size for var in ds_tmp]):

                    ## select data
                    xdata = ds_tmp['tas'].to_dataset(name='tas')
                    ydata = ds_tmp['lst']

                    ## fit
                    params_fit = get_best_fit(xdata, ydata, [f_lst], [params0], 
                        test_BIC0=True, 
                        print_report=False, file_name=name + f'__{mod_region}/tas_Reg_{mod}_{reg}')
                    
                    ## assign parameters
                    Par['a_Tl_Tg'].loc[{'reg_land': reg, 'model': mod}] = params_fit['a']
       

        ## STEP 2      
        ## precipitation

        ## make pattern scaling model
        def f_lsp(pr, lst, a1, a2):
            return a1 * pr + a2 * lst

        ## make parameters
        params1 = {'a1': dict(value=0, default=0),
            'a2': dict(value=0, default=0)}
        params2 = {'a1': dict(value=0, vary=False),
            'a2': dict(value=0, default=0)}                
        params3 = {'a1': dict(value=0, default=0),
            'a2': dict(value=0, vary=False)}

        ## loop on models and regions
        for mod in ds_delta.model.values:
            for reg in ds_delta.reg_land.values:
                print('\n', 'lsp', mod, reg, '\n')

                ## sub dataset and ignore empty regions
                ds_tmp = ds_delta.sel(reg_land=reg, model=mod).dropna('year', how='all')
                if all([ds_tmp[var].isnull().sum() < ds_tmp[var].size for var in ds_tmp]):

                    ## select data
                    xdata = ds_tmp.drop_vars([var for var in ds_tmp if var not in ['pr', 'lst']])
                    ydata = ds_tmp['lsp']

                    ## fit
                    params_fit = get_best_fit(xdata, ydata, [f_lsp], [params1, params2, params3], 
                        select_crit='BIC', test_BIC0=True, 
                        print_report=False, file_name=name + f'__{mod_region}/pr_Reg_{mod}_{reg}')
                    
                    ## assign parameters
                    Par['a_Pl_Pg'].loc[{'reg_land': reg, 'model': mod}] = params_fit['a1']
                    Par['a_Pl_Tl'].loc[{'reg_land': reg, 'model': mod}] = params_fit['a2']


        ## add units
        Par['a_Tl_Tg'].attrs['units'] = '1'
        Par['a_Pl_Pg'].attrs['units'] = '1'
        Par['a_Pl_Tl'].attrs['units'] = 'mm yr-1 K-1'

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

        ## make dimensions
        Par = Par.rename({'model': 'mod_clim'})

    ## return
    return Par.astype(np.float32)


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    if len(sys.argv) == 1:
        run_precalib(name, precalib_params, mod_region_list=[], regional=True)
    else:
        run_precalib(name, precalib_params, mod_region_list=[sys.argv[1]], regional=True)

