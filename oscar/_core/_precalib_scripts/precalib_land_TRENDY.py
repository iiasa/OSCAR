import sys
import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib, get_best_fit
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]
name = 'land_TRENDY'


##################################################
##   PRECALIBRATION OF LAND CARBON CYCLE
##################################################

##==========
## Ancillary
##==========

## information on PFTs
aggreg_pft = {'BareSoil': 'NonVeg', 
    'NonVeg': 'NonVeg', 
    'Crop': 'Crop', 
    'Grass': 'NonTree', 
    'ManGrass': 'ManGrass', 
    'Peat': 'NonTree', 
    'Shrub': 'NonTree', 
    'Tree': 'Tree'}

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

        ## OPTION:
        ## starting year of fits
        year_start_fit = 1921 # recycled climate before
        year_start_fit_ORCHIDEE = 1981 # weird CO2-fert before
        ## min number of years
        min_years = 31

        ## get original data
        with xr.open_dataset(path_precalib_in + f'{name}.nc') as TMP:
            ds = aggreg_regions(TMP, mod_region, weight_var={'lai_pft': 'area_pft', 'tas': '_area', 'pr': '_area'})
            ds = ds.drop_vars('_area')
            ds = ds.compute()

        ## split data
        ds_pft = ds.drop_vars([var for var in ds if 'PFT' not in ds[var].dims])
        ds_nopft = ds.drop_vars([var for var in ds if 'PFT' in ds[var].dims])

        ## ASSUMPTION:
        ## gap-fill Crop and ManGrass with Grass, set BareSoil and NonVeg to zero
        for var in ds_pft:
            ds_pft[var].loc[{'PFT': 'Crop'}] = ds_pft[var].sel(PFT='Crop').where(ds_pft['area_pft'].sel(PFT='Crop').notnull(), ds_pft[var].sel(PFT='Grass', drop=True))
            ds_pft[var].loc[{'PFT': 'ManGrass'}] = ds_pft[var].sel(PFT='ManGrass').where(ds_pft['area_pft'].sel(PFT='ManGrass').notnull(), ds_pft[var].sel(PFT='Grass', drop=True))
            ds_pft[var].loc[{'PFT': 'BareSoil'}] = 0 * ds_pft[var].sel(PFT='BareSoil')
            ds_pft[var].loc[{'PFT': 'NonVeg'}] = 0 * ds_pft[var].sel(PFT='NonVeg')

        ## ASSUMPTION:
        ## all fHarvest and fGrazing are on Crop and ManGrass, respectively
        ds_pft['fHarvest_pft'] = ds_nopft.fHarvest * xr.DataArray([bio == 'Crop' for bio in CWT.PFT.values], coords=CWT.PFT.coords)
        ds_pft['fGrazing_pft'] = ds_nopft.fGrazing * xr.DataArray([bio == 'ManGrass' for bio in CWT.PFT.values], coords=CWT.PFT.coords)
        ds_nopft = ds_nopft.drop_vars(['fHarvest', 'fGrazing'])

        ## ASSUMPTION:
        ## fHarvest and fGrazing cannot be more than 90% NPP
        ds_pft['fHarvest_pft'] = np.minimum(ds_pft.fHarvest_pft, 0.90 * ds_pft.npp_pft)
        ds_pft['fGrazing_pft'] = np.minimum(ds_pft.fGrazing_pft, 0.90 * ds_pft.npp_pft)

        ## further aggregate on PFTs
        ds_pft.coords['PFT2'] = xr.DataArray([aggreg_pft[bio] for bio in ds_pft['PFT'].values], dims='PFT')
        ds_pft['lai_pft'] *= ds_pft.area_pft
        ds_pft = ds_pft.groupby('PFT2').sum('PFT', min_count=1)
        ds_pft = ds_pft.rename({'PFT2': 'PFT'})
        ds_pft['lai_pft'] /= ds_pft.area_pft

        ## align to OSCAR biomes
        ds_pft = (CWT.pft_to_biome * ds_pft).sum('PFT', min_count=1)

        ## create complementary variables
        ## total mortality flux = cVeg loss other than fFire, fHarvest, fGrazing
        ds_pft['fMort_pft'] = ds_pft.npp_pft - ds_pft.fFire_pft.fillna(0.) - ds_pft.cVeg_pft.differentiate('year')
        ds_pft['fMort_pft'] -= ds_pft.fHarvest_pft.fillna(0.)
        ds_pft['fMort_pft'] -= ds_pft.fGrazing_pft.fillna(0.)
        

        ## STEP 1
        ## preindustrial steady state

        ## initialization
        Par = xr.Dataset()

        ## convenience function to define PI
        get_pi = lambda da: da.sel(exp='S0', drop=True).isel(year=slice(-50, None)).mean('year')

        ## preindustrial conditions
        Par['CO2_land'] = ds_nopft.co2.sel(exp='S0', drop=True).isel(year=0, drop=True).values
        Par['Tl_pi'] = ds_nopft.tas.sel(exp='S0', drop=True).isel(year=slice(None, 20)).mean('year')
        Par['Pl_pi'] = ds_nopft.pr.sel(exp='S0', drop=True).isel(year=slice(None, 20)).mean('year')

        ## core carbon cycle
        Par['npp_pi2'] = get_pi(ds_pft.npp_pft) / get_pi(ds_pft.area_pft)
        Par['v_mort'] = get_pi(ds_pft.fMort_pft) / get_pi(ds_pft.cVeg_pft)
        Par['v_resp'] = get_pi(ds_pft.rh_pft) / (get_pi(ds_pft.cLitter_pft) + get_pi(ds_pft.cSoil_pft))
        Par = Par.sel(model=Par.model[~ds_pft.cSoil_pft.isnull().all(['year', 'reg_land', 'bio_land', 'exp'])]).rename({'model': 'mod_Cland'})

        ## fire disturbance
        Par['v_fire'] = (get_pi(ds_pft.fFire_pft) / get_pi(ds_pft.cVeg_pft)).rename({'model': 'mod_Efire'}).dropna('mod_Efire', how='all')

        ## anthropogenic disturbances        
        Par['p_harv'] = (get_pi(ds_pft.fHarvest_pft) / get_pi(ds_pft.npp_pft)).rename({'model': 'mod_Eharv'}).dropna('mod_Eharv', how='all')
        Par['p_graz'] = (get_pi(ds_pft.fGrazing_pft) / get_pi(ds_pft.npp_pft)).rename({'model': 'mod_Egraz'}).dropna('mod_Egraz', how='all')

        ## convenience dataset for fits
        ## variables
        ds_fit = xr.Dataset()
        ds_fit.coords['model'] = ds_pft.model
        ds_fit['r_npp'] = ds_pft.npp_pft / ds_pft.area_pft / Par.npp_pi2.rename({'mod_Cland': 'model'})
        ds_fit['r_vFire'] = ds_pft.fFire_pft / ds_pft.cVeg_pft / Par.v_fire.rename({'mod_Efire': 'model'})
        ds_fit['r_vMort'] = ds_pft.fMort_pft / ds_pft.cVeg_pft / Par.v_mort.rename({'mod_Cland': 'model'})
        ds_fit['r_vResp'] = ds_pft.rh_pft / (ds_pft.cLitter_pft + ds_pft.cSoil_pft) / Par.v_resp.rename({'mod_Cland': 'model'})
        ds_fit['r_lai'] = ds_pft.lai_pft / get_pi(ds_pft.lai_pft)
        ds_fit['r_fMort'] = ds_pft.fMort_pft / get_pi(ds_pft.fMort_pft)
        ## sanitizing
        ds_fit = ds_fit.where(ds_fit > 0)
        ## drivers
        ds_fit['d_co2'] = ds_nopft.co2 - Par.CO2_land
        ds_fit['d_tas'] = ds_nopft.tas - Par.Tl_pi
        ds_fit['d_pr'] = ds_nopft.pr - Par.Pl_pi


        ## STEP 2
        ## transient NPP

        ## initialization of parameters
        for var in ['b_npp_CO2', 'x_npp_CO2', 'g_npp_T2', 'Topt_npp', 'x_npp_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2 only)
        def f_npp_co2(d_co2, co2_pi, bC, xC, pC):
            return 1 + bC * np.log1p(d_co2 / co2_pi) if xC==0 else 1 + bC/xC * ((1 + d_co2 / co2_pi)**xC - 1)
            #return 1 + bC/xC * ((1 + d_co2 / co2_pi)**xC - 1) / (1 + pC * ((1 + d_co2 / co2_pi)**xC - 1))

        ## make function (full)
        def f_npp(d_co2, d_tas, d_pr, co2_pi, bC, xC, pC, gT2, dTopt, pr_pi, xP):
            f_co2 = 1 + bC * np.log1p(d_co2 / co2_pi) if xC==0 else 1 + bC/xC * ((1 + d_co2 / co2_pi)**xC - 1)
            #f_co2 = 1 + bC/xC * ((1 + d_co2 / co2_pi)**xC - 1) / (1 + pC * ((1 + d_co2 / co2_pi)**xC - 1))
            f_tas = np.exp(-gT2 * (d_tas**2 - 2 * dTopt * d_tas))
            f_pr = (1 + d_pr / pr_pi) ** xP
            return  f_co2 * f_tas * f_pr

        ## loop on models and regions
        for mod in Par['mod_Cland'].values:
            for reg in Par.reg_land.values:
                for bio in Par.bio_land.values:
                    print('\n', 'npp', mod, reg, bio, '\n')

                    ## sub dataset and ignore empty regions
                    ds_tmp = ds_fit.drop_vars([var for var in ds_fit if var not in ['r_npp', 'd_co2', 'd_tas', 'd_pr']])
                    ds_tmp = ds_tmp.sel(reg_land=reg, bio_land=bio, model=mod).dropna('year', how='any')
                    ds_tmp = ds_tmp.sel(year=slice(year_start_fit_ORCHIDEE if mod == 'ORCHIDEE' else year_start_fit, None))
                    if len(ds_tmp.year) >= min_years:

                        ## STEP 2a
                        ## make parameters (CO2)
                        params1_npp_co2 = {'co2_pi': dict(value=Par.CO2_land.values, vary=False),
                            'bC': dict(value=0.65, min=0, default=0),
                            'xC': dict(value=-1, max=1, default=1),
                            'pC': dict(value=0, vary=False)}
                        params2_npp_co2 = {'co2_pi': dict(value=Par.CO2_land.values, vary=False),
                            'bC': dict(value=0.65, min=0, default=0),
                            'xC': dict(value=1, min=0, default=1),
                            'pC': dict(value=0, min=0, max=1, default=0)}

                        ## select data (CO2)
                        xdata = ds_tmp.d_co2.sel(exp='S1', drop=True).to_dataset(name='d_co2')
                        ydata = ds_tmp.r_npp.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_npp_co2], [params1_npp_co2], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/npp_co2_{mod}_{reg}_{bio}')

                        ## STEP 2b
                        ## make parameters
                        params1_npp = {'co2_pi': dict(value=Par.CO2_land.values, vary=False),
                            'bC': dict(value=params_fit['bC'], vary=False),
                            'xC': dict(value=params_fit['xC'], vary=False),
                            'pC': dict(value=params_fit['pC'], vary=False),
                            'gT2': dict(value=0.01, min=0, default=0), 
                            'dTopt': dict(value=0, min=0-(Par.Tl_pi.sel(reg_land=reg).values-273.15), max=30-(Par.Tl_pi.sel(reg_land=reg).values-273.15), default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0.5, max=1, default=0)}
                        params2_npp = {'co2_pi': dict(value=Par.CO2_land.values, vary=False),
                            'bC': dict(value=params_fit['bC'], vary=False),
                            'xC': dict(value=params_fit['xC'], vary=False),
                            'pC': dict(value=params_fit['pC'], vary=False),
                            'gT2': dict(value=0.01, min=0, default=0), 
                            'dTopt': dict(value=0, min=0-(Par.Tl_pi.sel(reg_land=reg).values-273.15), max=30-(Par.Tl_pi.sel(reg_land=reg).values-273.15), default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars('r_npp').sel(exp=['S1', 'S2'])
                        ydata = ds_tmp.r_npp.sel(exp=['S1', 'S2'])

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_npp], [params1_npp, params2_npp], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/npp_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['b_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['bC']
                        Par['x_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xC']
                        Par['g_npp_T2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT2']
                        Par['Topt_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['dTopt']
                        Par['x_npp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']

                    ## or assign default values
                    else:
                        Par['b_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 1.
                        Par['g_npp_T2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['Topt_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_npp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## STEP 3
        ## transient Fire

        ## initialization of parameters
        for var in ['x_fire_npp', 'g_fire_T', 'g_fire_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Efire']])

        ## make function (CO2 only)
        def f_fire_co2(r_npp, x):
            return r_npp**x

        ## make function (full)
        def f_fire(r_npp, d_tas, d_pr, x, gT, gP):
            f_co2 = r_npp**x
            f_tas = np.exp(gT * d_tas)
            f_pr = np.exp(gP * d_pr)
            return  f_co2 * f_tas * f_pr

        ## loop on models and regions
        for mod in Par['mod_Efire'].values:
            for reg in Par.reg_land.values:
                for bio in Par.bio_land.values:
                    print('\n', 'fire', mod, reg, bio, '\n')

                    ## sub dataset and ignore empty regions
                    ds_tmp = ds_fit.drop_vars([var for var in ds_fit if var not in ['r_vFire', 'r_npp', 'd_tas', 'd_pr']])
                    ds_tmp = ds_tmp.sel(reg_land=reg, bio_land=bio, model=mod).dropna('year', how='any')
                    ds_tmp = ds_tmp.sel(year=slice(year_start_fit, None))
                    if len(ds_tmp.year) >= min_years:

                        ## STEP 3a
                        ## make parameters (CO2)
                        params_fire_co2 = {'x': dict(value=1, default=0)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_npp.sel(exp='S1', drop=True).to_dataset(name='r_npp')
                        ydata = ds_tmp.r_vFire.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_fire_co2], [params_fire_co2], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/fire_co2_{mod}_{reg}_{bio}')

                        ## STEP 3b
                        ## make parameters
                        params1_fire = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'gP': dict(value=-0.1, max=0, default=0)}
                        params2_fire = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'gP': dict(value=0, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars('r_vFire').sel(exp=['S1', 'S2'])
                        ydata = ds_tmp.r_vFire.sel(exp=['S1', 'S2'])

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_fire], [params1_fire, params2_fire], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/fire_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_fire_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['x']
                        Par['g_fire_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['gT']
                        Par['g_fire_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['gP']

                    ## or assign default values
                    else:
                        Par['x_fire_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.
                        Par['g_fire_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.
                        Par['g_fire_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.


        ## STEP 4
        ## transient Mortality

        ## initialization of parameters
        for var in ['x_mort_npp', 'g_mort_T', 'x_mort_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2 only)
        def f_mort_co2(r_npp, x):
            return r_npp**x

        ## make function (full)
        def f_mort(r_npp, d_tas, d_pr, x, gT, pr_pi, xP):
            f_co2 = r_npp**x
            f_tas = np.exp(gT * d_tas)
            f_pr = (1 + d_pr / pr_pi) ** xP
            return  f_co2 * f_tas * f_pr

        ## loop on models and regions
        for mod in Par['mod_Cland'].values:
            for reg in Par.reg_land.values:
                for bio in Par.bio_land.values:
                    print('\n', 'mort', mod, reg, bio, '\n')

                    ## sub dataset and ignore empty regions
                    ds_tmp = ds_fit.drop_vars([var for var in ds_fit if var not in ['r_vMort', 'r_npp', 'd_tas', 'd_pr']])
                    ds_tmp = ds_tmp.sel(reg_land=reg, bio_land=bio, model=mod).dropna('year', how='any')
                    ds_tmp = ds_tmp.sel(year=slice(year_start_fit, None))
                    if len(ds_tmp.year) >= min_years:

                        ## STEP 4a
                        ## make parameters (CO2)
                        params_mort_co2 = {'x': dict(value=-1, max=0, default=0)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_npp.sel(exp='S1', drop=True).to_dataset(name='r_npp')
                        ydata = ds_tmp.r_vMort.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_mort_co2], [params_mort_co2], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/mort_co2_{mod}_{reg}_{bio}')

                        ## STEP 4b
                        ## make parameters
                        params1_mort = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=-0.1, max=0, default=0)}
                        params2_mort = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars('r_vMort').sel(exp=['S1', 'S2'])
                        ydata = ds_tmp.r_vMort.sel(exp=['S1', 'S2'])

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_mort], [params1_mort, params2_mort], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/mort_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_mort_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['x']
                        Par['g_mort_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT']
                        Par['x_mort_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']
                    
                    ## or assign default values
                    else:
                        Par['x_mort_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['g_mort_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_mort_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## STEP 5
        ## transient Respiration

        ## initialization of parameters
        for var in ['x_resp_fall', 'g_resp_T', 'x_resp_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2 only)
        def f_resp_co2(r_in, x):
            return r_in**x

        ## make function
        def f_resp(r_in, d_tas, d_pr, x, gT, pr_pi, xP):
            f_co2 = r_in**x
            f_tas = np.exp(gT * d_tas)
            f_pr = (1 + d_pr / pr_pi) ** xP
            return  f_co2 * f_tas * f_pr

        ## loop on models and regions
        for mod in Par['mod_Cland'].values:
            for reg in Par.reg_land.values:
                for bio in Par.bio_land.values:
                    print('\n', 'resp', mod, reg, bio, '\n')

                    ## sub dataset and ignore empty regions
                    ds_tmp = ds_fit.drop_vars([var for var in ds_fit if var not in ['r_vResp', 'r_fMort', 'd_tas', 'd_pr']])
                    ds_tmp = ds_tmp.sel(reg_land=reg, bio_land=bio, model=mod).dropna('year', how='any')
                    ds_tmp = ds_tmp.sel(year=slice(year_start_fit, None))
                    if len(ds_tmp.year) >= min_years:

                        ## STEP 5a
                        ## make parameters (CO2)
                        params_resp_co2 = {'x': dict(value=1, min=0, default=0)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_fMort.sel(exp='S1', drop=True).to_dataset(name='r_in')
                        ydata = ds_tmp.r_vResp.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_resp_co2], [params_resp_co2], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/resp_co2_{mod}_{reg}_{bio}')

                        ## STEP 5b
                        ## make parameters
                        params1_resp = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0.1, default=0)}
                        params2_resp = {'x': dict(value=params_fit['x'], vary=False),
                            'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_pi.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars('r_vResp').sel(exp=['S1', 'S2']).rename({'r_fMort': 'r_in'})
                        ydata = ds_tmp.r_vResp.sel(exp=['S1', 'S2'])

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_resp], [params1_resp, params2_resp], 
                            select_crit='BIC', print_report=False, file_name=name + f'__{mod_region}/resp_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_resp_fall'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['x']
                        Par['g_resp_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT']
                        Par['x_resp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']

                    ## or assign default values
                    else:
                        Par['x_resp_fall'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['g_resp_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_resp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## add units
        Par['CO2_land'].attrs['units'] = 'ppm'
        Par['npp_pi2'].attrs['units'] = 'PgC Mha-1 yr-1'
        for var in ['v_mort', 'v_resp', 'v_fire']:
            Par[var].attrs['units'] = 'yr-1'
        for var in ['p_harv', 'p_graz']:
            Par[var].attrs['units'] = '1'
        for var in ['b_npp_CO2', 'x_npp_CO2', 'x_fire_npp', 'x_mort_npp', 'x_resp_fall', 'x_npp_P', 'x_mort_P', 'x_resp_P']:
            Par[var].attrs['units'] = '1'
        for var in ['g_fire_T', 'g_mort_T', 'g_resp_T']:
            Par[var].attrs['units'] = 'K-1'
        Par['g_npp_T2'].attrs['units'] = 'K-2'
        Par['Topt_npp'].attrs['units'] = 'K'
        Par['g_fire_P'].attrs['units'] = 'yr mm-1'

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
    if len(sys.argv) == 1:
        run_precalib(name, precalib_params, mod_region_list=[], regional=True)
    else:
        run_precalib(name, precalib_params, mod_region_list=[sys.argv[1]], regional=True)

