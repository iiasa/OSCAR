import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'land_ISIMIP3a'


##################################################
##   PRECALIBRATION OF LAND CARBON SUBPOOLS
##################################################

##==========
## Ancillary
##==========

## information on PFTs
aggreg_pft = {'BareSoil': 'NonVeg', 
    'Crop': 'Crop', 
    'Grass': 'NonTree', 
    'ManGrass': 'ManGrass', 
    'Peat': 'NonTree', 
    'Shrub': 'NonTree', 
    'Tree': 'Tree'}

## PFT-to-biome crosswalk table (for now)
biome_list = ['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban']
pft_to_biome = {'Tree': 'Forest', 'NonTree': 'Non-Forest', 'Crop': 'Cropland', 'ManGrass': 'Pasture', 'NonVeg': 'Urban'}


##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        with xr.open_dataset(path_precalib_in + f'{name}.nc') as TMP:
            ds = TMP.isel(year=slice(-21, None)).mean('year')
            ds = aggreg_regions(ds, mod_region)
            ds = ds.compute()

        ## ASSUMPTION:
        ## gap-fill Crop and ManGrass with Grass, set BareSoil to zero
        for var in [var for var in ds if var != 'area'] + ['area']: # making sure area is last
            ds[var].loc[{'PFT': 'Crop'}] = ds[var].sel(PFT='Crop').where(ds['area'].sel(PFT='Crop').notnull(), ds[var].sel(PFT='Grass', drop=True))
            ds[var].loc[{'PFT': 'ManGrass'}] = ds[var].sel(PFT='ManGrass').where(ds['area'].sel(PFT='ManGrass').notnull(), ds[var].sel(PFT='Grass', drop=True))
            ds[var].loc[{'PFT': 'BareSoil'}] = 0. * ds[var].sel(PFT='BareSoil')
            
        ## further aggregation on PFTs
        ds.coords['PFT2'] = xr.DataArray([aggreg_pft[bio] for bio in ds['PFT'].values], dims='PFT')
        ds = ds.groupby('PFT2').sum('PFT', min_count=1, keep_attrs=True)
        ds = ds.rename({'PFT2': 'PFT'})

        ## align to OSCAR biomes
        ds = ds.rename({'PFT': 'bio_land'})
        ds.coords['bio_land'] = [pft_to_biome[bio] for bio in ds.bio_land.values]
        ds = ds.sel(bio_land=biome_list)

        ## initialization
        Par = xr.Dataset()

        ## vegetation subpools
        Par['p_wood'] = (ds.cWood / ds.cVeg).where(ds.cVeg > 0, 0.).rename({'model': 'mod_Cveg_part'})
        Par['p_root'] = (ds.cRoot / ds.cVeg).where(ds.cVeg > 0, 0.).rename({'model': 'mod_Cveg_part'})
        Par['p_leaf'] = (ds.cLeaf / ds.cVeg).where(ds.cVeg > 0, 0.).rename({'model': 'mod_Cveg_part'})
        p_sum = Par.p_wood + Par.p_root + Par.p_leaf
        Par['p_wood'] = Par['p_wood'].where(p_sum <= 1, Par['p_wood'] / p_sum)
        Par['p_root'] = Par['p_root'].where(p_sum <= 1, Par['p_root'] / p_sum)
        Par['p_leaf'] = Par['p_leaf'].where(p_sum <= 1, Par['p_leaf'] / p_sum)

        ## soil subpools
        cSoilTot = (ds.cLitter + ds.cSoil).where(ds.cLitter + ds.cSoil > 0).dropna('model', how='all')
        Par['p_litter'] = (ds.cLitter / cSoilTot).where(cSoilTot > 0, 0.).rename({'model': 'mod_Csoil_part'})
        Par['p_litter'] = np.minimum(Par['p_litter'], 1)

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

        ## add units
        for var in ['p_wood', 'p_root', 'p_leaf', 'p_litter']:
            Par[var].attrs['units'] = '1'

        ## format and make dimensions
        if 'exp' in Par.coords: Par = Par.drop_vars('exp')
        Par = Par.transpose('reg_land', 'bio_land', 'mod_Cveg_part', 'mod_Csoil_part')

    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

