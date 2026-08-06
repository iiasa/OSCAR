import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'land-albedo_Ouyang-2022'


##################################################
##   PRECALIBRATION OF ALBEDO EFFECT
##################################################

##==========
## Ancillary
##==========

## information on old biomes
old_axis = 'biome'
aggreg_biomes = {'Barren': 'Other', 
    'Cropland': 'Cropland', 
    'Forest': 'Forest', 
    'Grassland': 'Non-Forest', 
    'Ice': 'Other', 
    'Savanna': 'Non-Forest', 
    'Shrubland': 'Non-Forest', 
    'Urban': 'Urban', 
    'Water': 'Other', 
    'Wetland': 'Non-Forest'}
missing_biomes = {'Pasture': 'Non-Forest'}

## information on new biomes
new_axis = 'bio_land'
new_biomes_order = ['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban']


##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data on selected regions
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ds = aggreg_regions(TMP, mod_region)
            ds = ds.compute()

        ## further aggregate on biomes
        ds.coords[new_axis] = xr.DataArray([aggreg_biomes[bio] for bio in ds[old_axis].values], dims=old_axis)
        ds = ds.groupby(new_axis).sum(old_axis, min_count=1, keep_attrs=True)

        ## add missing biomes
        for bio in missing_biomes.keys():
            ds = xr.concat([ds, ds.sel({new_axis: missing_biomes[bio]}).assign_coords({new_axis: bio})], dim=new_axis)

        ## order biomes and axes
        ds = ds.sel({new_axis: new_biomes_order})
        ds = ds.transpose('reg_land', new_axis, 'LC')

        ## initialization
        Par = xr.Dataset()

        ## regional parameters
        Par['ph_lcc'] = ds.RFalb / ds.Area
        Par['ph_lcc'].attrs['units'] = 'W m-2 Mha-1' 

        ## ASSUMPTION: 
        ## choose actual LC first, and ideal LC second
        Par['ph_lcc'] = Par['ph_lcc'].sel(LC='actual', drop=True).combine_first(Par['ph_lcc'].sel(LC='ideal', drop=True))

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

