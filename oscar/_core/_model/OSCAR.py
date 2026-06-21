import importlib
import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


##################################################
##   OSCAR!
##################################################

## list of modules
list_modules = ['CO2', 'CH4', 'N2O', 'halo', 'H2', 
    'oceanC', 'landC', 
    'fire', 'wetland', 'permafrost', 
    'tropo', 'strato', 'NatEm',
    'ERF', 'clim']

## make model
OSCAR = Model('OSCAR')
for module in list_modules:
            
    ## import
    imported = importlib.import_module(f"oscar._core._model._modules.OSCAR_{module}")
    model = getattr(imported, f'OSCAR_{module}')
    
    ## merge
    OSCAR = OSCAR.merge(model, new_name=OSCAR.name)

