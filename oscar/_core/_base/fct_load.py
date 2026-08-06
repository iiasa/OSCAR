import os
import importlib
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths

path_data = get_paths()["core_data"]
path_precalib_out = get_paths()["params_precalib"]


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## convenience function to read parameters
def load_precalib_params(name, mod_region=None, xxx_global=False, xxx_global_ignore=[]):

    ## initialization
    Par = xr.Dataset()

    ## load global params
    if mod_region is None:
        with xr.open_dataset(path_precalib_out / f'{name}.nc') as TMP:
            for var in TMP: Par[var] = TMP[var].load()
    
    ## load regional params
    else:
        with xr.open_dataset(path_precalib_out / f'{name}__{mod_region}.nc') as TMP:
            for var in TMP: Par[var] = TMP[var].load()
    
    ## replace unknown land region with global if requested
    if mod_region is not None and xxx_global:
        with xr.open_dataset(path_precalib_out / f'{name}__Global.nc') as TMP:
            for var in [var for var in TMP if 'reg_land' in Par[var].dims and var not in xxx_global_ignore]:
                Par[var][{'reg_land': 0}] = TMP[var].sel({'reg_land': 'XXL'}, drop=True).load()

    ## return
    return Par


## general data loading function
def load_data(name):
    '''
    Convenience function to load any .nc dataset in the 'input_data' folder.
    
    Input:
    ------
    name (str)      part of file name -- must be unique to work
        
    Output:
    -------
    (xr.Dataset)    loaded dataset
    '''

    ## lists all available files
    list_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(path_data) for f in fn if f.endswith('.nc')]
    
    ## check compatible files
    okay_files = [f for f in list_files if name in f]

    ## error if no file or not unique
    if len(okay_files) == 0:
        raise RuntimeError("no files were found with input name: '{}')".format(name))
    elif len(okay_files) > 1:
        raise RuntimeError("more than one file was found:\n{}".format(okay_files))

    ## return loaded file otherwise
    with xr.open_dataset(okay_files[0]) as TMP: 
        return TMP.load()


##################################################
##   2. LOAD PARAMETERS
##################################################

## list of modules for parameters
list_modules = ['drivers', 
    'CO2', 'CH4', 'N2O', 'halo', 'H2', 
    'oceanC', 'landC_density', 'landC_bk', 
    'fire', 'wetland', 'permafrost', 
    'tropo', 'strato', 'NatEm',
    'ERF_wmghg', 'ERF_slcf', 'ERF_albedo', 'clim']

## load all primary parameters
def load_all_params(mod_region, get_drivers=True, list_modules=list_modules):

    ## initialize with constants
    imported = importlib.import_module(f"oscar._core._params_spec.Cst")
    Par = imported.get_params()

    ## loop on modules
    for module in [mod for mod in list_modules if mod != 'drivers' or get_drivers]:

            ## load
            imported = importlib.import_module(f"oscar._core._params_spec.Par_{module}")
            get_params = imported.get_params

            ## merge
            Par = xr.merge([Par, get_params(mod_region=mod_region)], join='outer', compat='no_conflicts')
            Par['reg_land'].attrs['mod_region'] = mod_region

    ## return
    return Par

