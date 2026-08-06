import sys
import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib, get_best_fit
from oscar._core._base.fct_regions import aggreg_regions
from oscar._core._base.fct_solve import safe_exp, f_max, safe_ratio

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

        ## OPTION:
        ## starting year of fits
        year_start_fit = 1921 # recycled climate before
        ## min number of years (otherwise skipped)
        min_years = 51

        ## DEFINITION:
        ## convenience function for preindustrial
        get_pi = lambda da: da.sel(exp='S0', drop=True).isel(year=slice(-40, None)).mean('year')

        ## get original data
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ds = aggreg_regions(TMP, mod_region, weight_var={'lai_pft': 'area_pft', 'tas': '_area', 'pr': '_area'})
            ds = ds.drop_vars('_area')
            ds = ds.compute()

        ## split data
        ds_pft = ds.drop_vars([var for var in ds if 'PFT' not in ds[var].dims])
        ds_nopft = ds.drop_vars([var for var in ds if 'PFT' in ds[var].dims])

        ## SANITIZATION:
        ## set to NaN if a key variable is negative or zero
        test_vars = [(get_pi(ds_pft[var+'_pft']) <= 0) for var in ['area', 'npp', 'rh', 'cVeg', 'cSoil']]
        ds_pft = ds_pft.where(~xr.concat(test_vars, dim="stack").any(dim="stack"))
        ds_pft['cLitter_pft'] = ds_pft.cLitter_pft.where(ds_pft.cLitter_pft > 0)

        ## ASSUMPTION:
        ## gap-fill Crop and ManGrass with Grass, set BareSoil and NonVeg to zero
        for var in [var for var in ds_pft if var != 'area_pft'] + ['area_pft']: # making sure area is last
            ds_pft[var].loc[{'PFT': 'Crop'}] = ds_pft[var].sel(PFT='Crop').where(ds_pft['area_pft'].sel(PFT='Crop').notnull(), ds_pft[var].sel(PFT='Grass', drop=True))
            ds_pft[var].loc[{'PFT': 'ManGrass'}] = ds_pft[var].sel(PFT='ManGrass').where(ds_pft['area_pft'].sel(PFT='ManGrass').notnull(), ds_pft[var].sel(PFT='Grass', drop=True))
            ds_pft[var].loc[{'PFT': 'BareSoil'}] = 0 * ds_pft[var].sel(PFT='BareSoil')
            ds_pft[var].loc[{'PFT': 'NonVeg'}] = 0 * ds_pft[var].sel(PFT='NonVeg')

        ## ASSUMPTION:
        ## take fHarvest and fGrazing only on Crop and ManGrass, respectively
        ds_pft['fHarvest_pft'] = (ds_nopft.fHarvest * (ds_pft.PFT == 'Crop')).where(ds_nopft.fHarvest.notnull(), 
            ds_nopft.fHarvest * xr.DataArray([bio == 'Crop' for bio in ds_pft.PFT.values], coords=ds_pft.PFT.coords))
        ds_pft['fGrazing_pft'] = (ds_nopft.fGrazing * (ds_pft.PFT == 'ManGrass')).where(ds_nopft.fHarvest.notnull(), 
            ds_nopft.fHarvest * xr.DataArray([bio == 'Crop' for bio in ds_pft.PFT.values], coords=ds_pft.PFT.coords))

        ## further aggregate on PFTs
        ds_pft.coords['PFT2'] = xr.DataArray([aggreg_pft[bio] for bio in ds_pft['PFT'].values], dims='PFT')
        ds_pft['lai_pft'] *= ds_pft.area_pft
        ds_pft = ds_pft.groupby('PFT2').sum('PFT', min_count=1).rename({'PFT2': 'PFT'})
        ds_pft = ds_pft.transpose(*[dim for dim in list(ds.dims)])
        ds_pft['lai_pft'] /= ds_pft.area_pft

        ## align to OSCAR biomes
        ds_pft = ds_pft.rename({'PFT': 'bio_land'})
        ds_pft.coords['bio_land'] = [pft_to_biome[bio] for bio in ds_pft.bio_land.values]
        ds_pft = ds_pft.sel(bio_land=biome_list)

        ## ASSUMPTION/SANITIZATION:
        ## extraction cannot be more than 90% NPP at steady state, reduce extraction
        fHarvest_clip = np.clip(get_pi(ds_pft.fHarvest_pft), 0., 0.90 * get_pi(ds_pft.npp_pft))
        ds_pft['fHarvest_pft'] *= fHarvest_clip / get_pi(ds_pft.fHarvest_pft)
        fGrazing_clip = np.clip(get_pi(ds_pft.fGrazing_pft), 0., 0.90 * get_pi(ds_pft.npp_pft))
        ds_pft['fGrazing_pft'] *= fGrazing_clip / get_pi(ds_pft.fGrazing_pft)

        ## ASSUMPTION/SANITIZATION:
        ## fire cannot be more than 2x cVeg at steady-state, reduce fire
        fFire_clip = np.clip(get_pi(ds_pft.fFire_pft), 0., 2. * get_pi(ds_pft.cVeg_pft))
        ds_pft['fFire_pft'] *= fFire_clip / get_pi(ds_pft.fFire_pft)

        ## ASSUMPTION/SANITIZATION:
        ## total loss cannot be more than 99% NPP at steady state, reduce fire
        ds_pft['fLoss_pft'] = ds_pft.fFire_pft.fillna(0.) + ds_pft.fHarvest_pft.fillna(0.) + ds_pft.fGrazing_pft.fillna(0.)
        fLoss_clip = np.clip(get_pi(ds_pft.fLoss_pft), 0., 0.99 * get_pi(ds_pft.npp_pft))
        fLoss_delta = get_pi(ds_pft.fLoss_pft) - fLoss_clip
        ds_pft['fFire_pft'] -= np.minimum(fLoss_delta, get_pi(ds_pft.fFire_pft))
        ds_pft['fLoss_pft'] = ds_pft.fFire_pft.fillna(0.) + ds_pft.fHarvest_pft.fillna(0.) + ds_pft.fGrazing_pft.fillna(0.)

        ## ASSUMPTION/SANITIZATION:
        ## mortality must be between 0.001x and 3x cVeg at steady-state, change npp
        ds_pft['fMort_pft'] = ds_pft.npp_pft - ds_pft.fLoss_pft - ds_pft.cVeg_pft.differentiate('year')
        fMort_clip = np.clip(get_pi(ds_pft.fMort_pft), 0.001 * get_pi(ds_pft.cVeg_pft), 3. * get_pi(ds_pft.cVeg_pft))
        fMort_delta = get_pi(ds_pft.fMort_pft) - fMort_clip
        ds_pft['npp_pft'] -= np.minimum(fMort_delta, get_pi(ds_pft.npp_pft))
        ds_pft['fMort_pft'] = ds_pft.npp_pft - ds_pft.fLoss_pft - ds_pft.cVeg_pft.differentiate('year')


        ## STEP 1
        ## preindustrial steady state

        ## initialization
        Par = xr.Dataset()

        ## preindustrial conditions
        Par['CO2_piL'] = ds_nopft.co2.sel(exp='S0', drop=True).isel(year=0, drop=True).values
        Par['Tl_piL'] = ds_nopft.tas.sel(exp='S0', drop=True).isel(year=slice(None, 20)).mean('year')
        Par['Pl_piL'] = ds_nopft.pr.sel(exp='S0', drop=True).isel(year=slice(None, 20)).mean('year')

        ## core carbon cycle
        Par['npp_piL'] = get_pi(ds_pft.npp_pft) / get_pi(ds_pft.area_pft)
        Par['v_mort'] = get_pi(ds_pft.fMort_pft) / get_pi(ds_pft.cVeg_pft)
        Par['v_resp'] = get_pi(ds_pft.rh_pft) / (get_pi(ds_pft.cLitter_pft).fillna(0.) + get_pi(ds_pft.cSoil_pft))
        Par = Par.sel(model=Par.model[~ds_pft.cSoil_pft.isnull().all(['year', 'reg_land', 'bio_land', 'exp'])]).rename({'model': 'mod_Cland'})
        not_null = Par['npp_piL'].notnull() & Par['v_mort'].notnull() & Par['v_resp'].notnull()
        for var in ['npp_piL', 'v_mort', 'v_resp']: Par[var] = Par[var].where(not_null, 0.)

        ## fire disturbance
        Par['v_fire'] = (get_pi(ds_pft.fFire_pft) / get_pi(ds_pft.cVeg_pft)).rename({'model': 'mod_Efire'}).dropna('mod_Efire', how='all').fillna(0.)

        ## anthropogenic disturbances        
        Par['p_charv'] = (get_pi(ds_pft.fHarvest_pft) / get_pi(ds_pft.npp_pft)).rename({'model': 'mod_Echarv'}).dropna('mod_Echarv', how='all').fillna(0.)
        Par['p_graz'] = (get_pi(ds_pft.fGrazing_pft) / get_pi(ds_pft.npp_pft)).rename({'model': 'mod_Egraz'}).dropna('mod_Egraz', how='all').fillna(0.)


        ## convenience dataset for fits
        ## correction of drift
        ds_fit = ds_pft - ds_pft.sel(exp='S0', drop=True) + get_pi(ds_pft)
        ## fit variables
        ds_fit['r_npp'] = ds_fit.npp_pft / ds_fit.area_pft / Par.npp_piL.rename({'mod_Cland': 'model'})
        ds_fit['r_vFire'] = ds_fit.fFire_pft / ds_fit.cVeg_pft / Par.v_fire.rename({'mod_Efire': 'model'})
        ds_fit['r_vMort'] = ds_fit.fMort_pft / ds_fit.cVeg_pft / Par.v_mort.rename({'mod_Cland': 'model'})
        ds_fit['r_vResp'] = ds_fit.rh_pft / (ds_fit.cLitter_pft.fillna(0.) + ds_fit.cSoil_pft) / Par.v_resp.rename({'mod_Cland': 'model'})
        ds_fit['r_lai'] = ds_fit.lai_pft / get_pi(ds_fit.lai_pft)
        ds_fit['r_fMort'] = ds_fit.fMort_pft / get_pi(ds_fit.fMort_pft)
        ## sanitizing
        ds_fit = ds_fit.where(np.isfinite(ds_fit))
        ## drivers
        ds_fit['d_co2'] = ds_nopft.co2 - Par.CO2_piL
        ds_fit['d_tas'] = ds_nopft.tas - Par.Tl_piL
        ds_fit['d_pr'] = ds_nopft.pr - Par.Pl_piL


        ## STEP 2
        ## transient NPP

        ## initialization of parameters
        for var in ['b_npp_CO2', 'x_npp_CO2', 'g_npp_T2', 'D_Topt_npp', 'x_npp_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2)
        def f_npp_co2(d_co2, co2_pi, bC, xC, pC):
            f_log = 1 + bC * np.log1p(d_co2 / co2_pi)
            f_full = 1 + bC/xC * ((1 + d_co2 / co2_pi)**xC - 1) / (1 + pC * ((1 + d_co2 / co2_pi)**xC - 1))
            return f_log if xC==0 else f_full

        ## make function (clim)
        def f_npp_clim(d_tas, d_pr, gT2, dTopt, pr_pi, xP):
            f_tas = safe_exp(gT2 * 2 * dTopt * d_tas, 100) * np.exp(-gT2 * d_tas**2)
            f_pr = safe_exp(xP * np.log(safe_ratio(1 + d_pr / pr_pi)), 100)
            return  f_tas * f_pr

        ## make function (full)
        def f_npp(d_co2, d_tas, d_pr, co2_pi, bC, xC, pC, gT2, dTopt, pr_pi, xP):
            f_co2 = f_npp_co2(d_co2, co2_pi, bC, xC, pC)
            f_clim = f_npp_clim(d_tas, d_pr, gT2, dTopt, pr_pi, xP)
            return  f_co2 * f_clim

        ## loop on models and regions
        for mod in Par['mod_Cland'].values:
            for reg in Par.reg_land.values:
                for bio in Par.bio_land.values:
                    print('\n', 'npp', mod, reg, bio, '\n')

                    ## sub dataset and ignore empty regions
                    ds_tmp = ds_fit.drop_vars([var for var in ds_fit if var not in ['r_npp', 'd_co2', 'd_tas', 'd_pr']])
                    ds_tmp = ds_tmp.sel(reg_land=reg, bio_land=bio, model=mod).dropna('year', how='any')
                    ds_tmp = ds_tmp.sel(year=slice(year_start_fit, None))
                    if len(ds_tmp.year) >= min_years:

                        ## STEP 2a
                        ## make parameters (CO2)
                        params1_npp_co2 = {'co2_pi': dict(value=Par.CO2_piL.values, vary=False),
                            'bC': dict(value=0.65, min=0, default=0),
                            'xC': dict(value=-1, max=1, default=1),
                            'pC': dict(value=0, vary=False)}
                        params2_npp_co2 = {'co2_pi': dict(value=Par.CO2_piL.values, vary=False),
                            'bC': dict(value=0.65, min=0, default=0),
                            'xC': dict(value=1, min=0, default=1),
                            'pC': dict(value=0.1, min=0, max=1, default=0)}

                        ## select data (CO2)
                        xdata = ds_tmp.d_co2.sel(exp='S1', drop=True).to_dataset(name='d_co2')
                        ydata = ds_tmp.r_npp.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_npp_co2], [params1_npp_co2], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/npp_co2_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['b_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['bC']
                        Par['x_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xC']
                        #Par['b_npp_sat'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['pC']

                        ## STEP 2b
                        ## make parameters
                        params1_npp_clim = {
                            'gT2': dict(value=0.01, min=0, default=0), 
                            'dTopt': dict(value=0, min=0-(Par.Tl_piL.sel(reg_land=reg).values-273.15), max=30-(Par.Tl_piL.sel(reg_land=reg).values-273.15), default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0.5, min=0, default=0)}
                        params2_npp_clim = {
                            'gT2': dict(value=0.01, min=0, default=0), 
                            'dTopt': dict(value=0, min=0-(Par.Tl_piL.sel(reg_land=reg).values-273.15), max=30-(Par.Tl_piL.sel(reg_land=reg).values-273.15), default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars(['r_npp', 'd_co2']).sel(exp='S2')
                        ydata = ds_tmp.r_npp.sel(exp='S2') / ds_tmp.r_npp.sel(exp='S1')

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_npp_clim], [params1_npp_clim, params2_npp_clim], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/npp_clim_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['g_npp_T2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT2']
                        Par['D_Topt_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['dTopt']
                        Par['x_npp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']

                    ## or assign default values
                    else:
                        Par['b_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_npp_CO2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 1.
                        #Par['b_npp_sat'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['g_npp_T2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['D_Topt_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_npp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## STEP 3
        ## transient Fire

        ## initialization of parameters
        for var in ['x_fire_npp', 'x_fire_npp2', 'g_fire_T', 'g_fire_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Efire']])

        ## make function (CO2)
        def f_fire_co2(r_npp, x, x2, v):
            f_npp = safe_exp(x * np.log(safe_ratio(r_npp)), f_max(v))
            f_npp2 = safe_exp(x2 * np.log(safe_ratio(r_npp))**2, f_max(v))
            return f_npp * f_npp2

        ## make function (clim)
        def f_fire_clim(d_tas, d_pr, gT, gP, v):
            f_tas = safe_exp(gT * d_tas, f_max(v))
            f_pr = safe_exp(gP * d_pr, f_max(v))
            return  f_tas * f_pr

        ## make function (full)
        def f_fire(r_npp, d_tas, d_pr, x, x2, gT, gP, v):
            f_co2 = f_fire_co2(r_npp, x, x2, v)
            f_tas = f_fire_clim(d_tas, d_pr, gT, gP, v)
            return  f_co2 * f_clim

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
                        params1_fire_co2 = {'x': dict(value=0, default=0),
                            'x2': dict(value=-0.01, max=0, default=0),
                            'v': dict(value=Par.v_fire.sel(mod_Efire=mod, reg_land=reg, bio_land=bio).values, vary=False)}
                        params2_fire_co2 = {'x': dict(value=0, default=0),
                            'x2': dict(value=0, vary=False),
                            'v': dict(value=Par.v_fire.sel(mod_Efire=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_npp.sel(exp='S1', drop=True).to_dataset(name='r_npp')
                        ydata = ds_tmp.r_vFire.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_fire_co2], [params1_fire_co2, params2_fire_co2], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/fire_co2_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_fire_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['x']
                        Par['x_fire_npp2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['x2']

                        ## STEP 3b
                        ## make parameters
                        params1_fire_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'gP': dict(value=-0.1, max=0, default=0),
                            'v': dict(value=Par.v_fire.sel(mod_Efire=mod, reg_land=reg, bio_land=bio).values, vary=False)}
                        params2_fire_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'gP': dict(value=0, vary=False),
                            'v': dict(value=Par.v_fire.sel(mod_Efire=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars(['r_vFire', 'r_npp']).sel(exp='S2')
                        ydata = ds_tmp.r_vFire.sel(exp='S2') / ds_tmp.r_vFire.sel(exp='S1')

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_fire_clim], [params1_fire_clim, params2_fire_clim], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/fire_clim_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['g_fire_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['gT']
                        Par['g_fire_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = params_fit['gP']

                    ## or assign default values
                    else:
                        Par['x_fire_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.
                        Par['x_fire_npp2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.
                        Par['g_fire_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.
                        Par['g_fire_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Efire': mod}] = 0.


        ## STEP 4
        ## transient Mortality

        ## initialization of parameters
        for var in ['x_mort_npp', 'g_mort_T', 'x_mort_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2)
        def f_mort_co2(r_npp, x, x2, v):
            f_npp = safe_exp(x * np.log(safe_ratio(r_npp)), f_max(v))
            f_npp2 = safe_exp(x2 * np.log(safe_ratio(r_npp))**2, f_max(v))
            return f_npp * f_npp2

        ## make function (clim)
        def f_mort_clim(d_tas, d_pr, gT, pr_pi, xP, v):
            f_tas = safe_exp(gT * d_tas, f_max(v))
            f_pr = safe_exp(xP * np.log(safe_ratio(1 + d_pr / pr_pi)), f_max(v))
            return  f_tas * f_pr

        ## make function (full)
        def f_mort(r_npp, d_tas, d_pr, x, gT, pr_pi, xP, v):
            f_co2(r_npp, x, x2, v)
            f_clim(d_tas, d_pr, gT, pr_pi, xP, v)
            return  f_co2 * f_clim

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
                        params1_mort_co2 = {'x': dict(value=0, default=0),
                            'x2': dict(value=-0.01, max=0, default=0),
                            'v': dict(value=Par.v_mort.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}
                        params2_mort_co2 = {'x': dict(value=0, default=0),
                            'x2': dict(value=0, vary=False),
                            'v': dict(value=Par.v_mort.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_npp.sel(exp='S1', drop=True).to_dataset(name='r_npp')
                        ydata = ds_tmp.r_vMort.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_mort_co2], [params2_mort_co2], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/mort_co2_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_mort_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['x']
                        #Par['x_mort_npp2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['x2']

                        ## STEP 4b
                        ## make parameters
                        params1_mort_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=-0.1, max=0, default=0),
                            'v': dict(value=Par.v_mort.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}
                        params2_mort_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False),
                            'v': dict(value=Par.v_mort.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars(['r_vMort', 'r_npp']).sel(exp='S2')
                        ydata = ds_tmp.r_vMort.sel(exp='S2') / ds_tmp.r_vMort.sel(exp='S1')

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_mort_clim], [params1_mort_clim, params2_mort_clim], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/mort_clim_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['g_mort_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT']
                        Par['x_mort_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']
                    
                    ## or assign default values
                    else:
                        Par['x_mort_npp'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        #Par['x_mort_npp2'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['g_mort_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_mort_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## STEP 5
        ## transient Respiration

        ## initialization of parameters
        for var in ['x_resp_fall', 'g_resp_T', 'x_resp_P']:
            Par[var] = np.nan + sum([xr.zeros_like(Par[dim], dtype=np.float32) for dim in ['reg_land', 'bio_land', 'mod_Cland']])

        ## make function (CO2)
        def f_resp_co2(r_in, x, v):
            return safe_exp(x * np.log(safe_ratio(r_in)), f_max(v))

        ## make function (clim)
        def f_resp_clim(d_tas, d_pr, gT, pr_pi, xP, v):
            f_tas = safe_exp(gT * d_tas, f_max(v))
            f_pr = safe_exp(xP * np.log(safe_ratio(1 + d_pr / pr_pi)), f_max(v))
            return  f_tas * f_pr

        ## make function (full)
        def f_resp(r_in, d_tas, d_pr, x, gT, pr_pi, xP, v):
            f_co2 = safe_exp(x * np.log(safe_ratio(r_in)), f_max(v))
            f_clim(d_tas, d_pr, gT, pr_pi, xP, v)
            return  f_co2 * f_clim

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
                        params_resp_co2 = {'x': dict(value=0.1, min=0, max=1, default=0),
                            'v': dict(value=Par.v_resp.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data (CO2)
                        xdata = ds_tmp.r_fMort.sel(exp='S1', drop=True).to_dataset(name='r_in')
                        ydata = ds_tmp.r_vResp.sel(exp='S1', drop=True)

                        ## fit (CO2)
                        params_fit = get_best_fit(xdata, ydata, [f_resp_co2], [params_resp_co2], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/resp_co2_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['x_resp_fall'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['x']

                        ## STEP 5b
                        ## make parameters
                        params1_resp_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0.1, default=0),
                            'v': dict(value=Par.v_resp.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}
                        params2_resp_clim = {'gT': dict(value=0.1, min=0, default=0), 
                            'pr_pi': dict(value=Par.Pl_piL.sel(reg_land=reg).values, vary=False),
                            'xP': dict(value=0, vary=False),
                            'v': dict(value=Par.v_resp.sel(mod_Cland=mod, reg_land=reg, bio_land=bio).values, vary=False)}

                        ## select data
                        xdata = ds_tmp.drop_vars(['r_vResp', 'r_fMort']).sel(exp='S2')
                        ydata = ds_tmp.r_vResp.sel(exp='S2') / ds_tmp.r_vResp.sel(exp='S1')

                        ## fit
                        params_fit = get_best_fit(xdata, ydata, [f_resp_clim], [params1_resp_clim, params2_resp_clim], 
                            select_crit='BIC', test_BIC1=True, 
                            print_report=False, file_name=name + f'__{mod_region}/resp_clim_{mod}_{reg}_{bio}')

                        ## assign parameters
                        Par['g_resp_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['gT']
                        Par['x_resp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = params_fit['xP']

                    ## or assign default values
                    else:
                        Par['x_resp_fall'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['g_resp_T'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.
                        Par['x_resp_P'].loc[{'reg_land': reg, 'bio_land': bio, 'mod_Cland': mod}] = 0.


        ## add units
        Par['CO2_piL'].attrs['units'] = 'ppm'
        Par['Tl_piL'].attrs['units'] = 'K'
        Par['Pl_piL'].attrs['units'] = 'mm yr-1'
        Par['npp_piL'].attrs['units'] = 'PgC Mha-1 yr-1'
        for var in ['v_mort', 'v_resp', 'v_fire']:
            Par[var].attrs['units'] = 'yr-1'
        for var in ['p_charv', 'p_graz']:
            Par[var].attrs['units'] = '1'
        for var in ['b_npp_CO2', 'x_npp_CO2', 'x_fire_npp', 'x_fire_npp2', 'x_mort_npp', 'x_resp_fall', 'x_npp_P', 'x_mort_P', 'x_resp_P']:
            Par[var].attrs['units'] = '1'
        for var in ['g_fire_T', 'g_mort_T', 'g_resp_T']:
            Par[var].attrs['units'] = 'K-1'
        Par['g_npp_T2'].attrs['units'] = 'K-2'
        Par['D_Topt_npp'].attrs['units'] = 'K'
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

