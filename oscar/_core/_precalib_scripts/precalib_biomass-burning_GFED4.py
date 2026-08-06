import warnings
import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_precalib import run_precalib
from oscar._core._base.fct_regions import aggreg_regions


name = 'biomass-burning_GFED4'


##################################################
##   PRECALIBRATION OF LAND CARBON SUBPOOLS
##################################################

##==========
## Ancillary
##==========

## information on PFTs
## note: peat ignored in this version of OSCAR
sect_pft = {'gra': 'NonTree', 
    'awb': 'Crop', 
    'bor': 'Tree', 
    'tem': 'Tree', 
    'def': 'Tree',
    'pea': 'NonVeg'}

## PFT-to-biome crosswalk table
## note: for now, simple arbitrary assumptions
CWT = xr.Dataset()
CWT.coords['PFT'] = ['NonVeg', 'Tree', 'NonTree', 'Crop', 'ManGrass']
CWT.coords['bio_land'] = ['Forest', 'Non-Forest', 'Cropland', 'Pasture', 'Urban']
CWT['pft_to_biome'] = (('bio_land', 'PFT'), [[0.05, 0.80, 0.15, 0.00, 0.00], # Forest
                                          [0.10, 0.10, 0.80, 0.00, 0.00], # Non-Forest
                                          [0.10, 0.05, 0.05, 0.70, 0.10], # Cropland
                                          [0.10, 0.05, 0.35, 0.00, 0.50], # Pasture
                                          [0.95, 0.01, 0.04, 0.00, 0.00]]) # Urban


##=========
## Function
##=========

## precalibration function
def precalib_params(mod_region, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        ds = load_data('biomass-burning_GFED4.nc')
        ds = ds.drop_vars('Aburn').mean('year')            
        ds = aggreg_regions(ds, mod_region)
        ds = ds.compute()

        ## turn sectors into PFTs
        ds.coords['PFT'] = xr.DataArray([sect_pft[sec] for sec in ds.sect.values], dims='sect')
        ds = ds.groupby('PFT').sum('sect', min_count=1, keep_attrs=True)

        ## ASSUMPTION:
        ## make ManGrass as NonTree, NonVeg as zero
        ds = xr.concat([ds, ds.sel(PFT='NonTree').assign_coords(PFT='ManGrass')], dim='PFT')
        ds = ds.where(ds.PFT != 'pea', 0.)

        ## align to OSCAR biomes
        ds =  (CWT.pft_to_biome * ds).sum('PFT', min_count=1)

        ## initialization
        Par = xr.Dataset()

        ## emission factors
        for spc in ['CH4', 'N2O', 'H2', 'BC', 'OC', 'SO2', 'NH3', 'NOx', 'CO', 'VOC']:
            Par['a_bb_'+spc] = ds['Ebb_'+spc] / ds['Ebb_C']
            Par['a_bb_'+spc] =  Par['a_bb_'+spc].where(ds['Ebb_C'] != 0, 0.)

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

        ## add units
        for spc in ['CH4', 'BC', 'OC', 'CO']:
            Par['a_bb_'+spc].attrs['units'] = 'TgC PgC-1'
        for spc in ['N2O', 'NH3', 'NOx']:
            Par['a_bb_'+spc].attrs['units'] = 'TgN PgC-1'
        Par['a_bb_'+'H2'].attrs['units'] = 'TgH2 PgC-1'
        Par['a_bb_'+'SO2'].attrs['units'] = 'TgS PgC-1'
        Par['a_bb_'+'VOC'].attrs['units'] = 'Tg PgC-1'

        ## format
        Par = Par.transpose('reg_land', 'bio_land')

    ## return
    return Par


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=True)

