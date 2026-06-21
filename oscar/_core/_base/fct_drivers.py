import numpy as np
import xarray as xr


##################################################
##   1. GENERIC FORMATTING
##################################################

## function to format LUH for OSCAR
def format_LUH(ds_in, 
    year_pi=1750, year_pd=2022, 
    scen_hist='hist', keep_harv_dC=False):

    ## initialize as of PI 
    For = ds_in.sel(year=slice(year_pi, None)).copy(deep=True)
    if 'scen' in ds_in.dims: For_hist = For.sel(scen=scen_hist, drop=True).dropna('year', how='all')
    else: For_hist = For.dropna('year', how='all')

    ## get long-term historical for age
    if 'scen' in ds_in.dims: For_age = ds_in.sel(scen=scen_hist, drop=True).dropna('year', how='all')
    else: For_age = ds_in.dropna('year', how='all')

    ## PI land cover
    For['Aland_pi'] = (For_hist.Aland1 + For_hist.Aland2).sel(year=year_pi)
    For['Aland_bk_pi'] = For_hist.Aland2.sel(year=year_pi)
    For['Aland_pd'] = (For_hist.Aland1 + For_hist.Aland2).sel(year=year_pd) if year_pd in For_hist.year else np.nan * For.Aland_pi
    for var in ['Aland_pi', 'Aland_bk_pi', 'Aland_pd']:
        For[var].attrs['units'] = For_hist.Aland1.units
        For[var].attrs['year'] = year_pi if var.endswith('_pi') else year_pd
    For = For.drop_vars(['Aland1', 'Aland2'])

    ## PI land use
    For['dC_wharv_pi'] = For_hist.dC_wharv2.sel(year=year_pi)
    For['dA_wharv_pi'] = For_hist.dA_wharv2.sel(year=year_pi)
    For['dA_shift_pi'] = For_hist.dA_shift.sel(year=year_pi)
    for var in ['dC_wharv_pi', 'dA_wharv_pi', 'dA_shift_pi']:
        For[var].attrs['units'] = For_hist.dC_wharv2.units if var.startswith('dC') else For_hist.dA_wharv2.units
        For[var].attrs['year'] = year_pi


    ## Delta forcings
    for var in [var for var in For if not var.endswith('_pi') and not var.endswith('_pd')]:
        if 'lcc' in var or 'harv1' in var:
            For['D_'+var] = For[var]
            For['D_'+var].attrs['units'] = For[var].units
            For = For.drop_vars(var)
        else:
            For['D_'+var] = For[var] - For_hist[var].sel(year=year_pi, drop=True)
            For['D_'+var].attrs['units'] = For[var].units
            For = For.drop_vars(var)


    ## mean age of secondary land
    ## secondary land (excl. shifting)
    Aland2 = For_age.dA_lcc1.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'}).cumsum('year')
    Aland2 += For_age.dA_lcc2.sum('bio_from', min_count=1).rename({'bio_to':'bio_land'}).cumsum('year') - For_age.dA_lcc2.sum('bio_to', min_count=1).rename({'bio_from':'bio_land'}).cumsum('year')
    Aland2 += For_age.dA_wharv1.cumsum('year')
    ## loss & gain
    loss = For_age.dA_lcc2.sum('bio_to', min_count=1).rename({'bio_from':'bio_land'})
    loss += For_age.dA_wharv2
    gain = (For_age.dA_lcc1 + For_age.dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    gain += (For_age.dA_wharv1 + For_age.dA_wharv2)
    ## survival rate
    years = For_age.year.sel(year=slice(None, year_pi-1))
    survival_pi = xr.concat([np.maximum((1 - loss / Aland2).where(Aland2 > 0, 1.), 0.).sel(year=slice(yr, year_pi-1)).prod('year').assign_coords(year=yr) for yr in years.values], dim='year')
    ## mean age
    age_pi = (year_pi - years).weighted((gain * survival_pi).fillna(0.)).mean('year')
    age_max = np.max(year_pi - years)
    age_pi = age_pi.where(age_pi > 0).where(age_pi < age_max).fillna(0.5*age_max)
    For['age_bk_pi'] = age_pi
    For['age_bk_pi'].attrs['units'] = 'yr'
    For['age_bk_pi'].attrs['year'] = year_pi


    ## return 
    if not keep_harv_dC: For = For.drop_vars([var for var in For if 'dC_wharv' in var])
    return For


## function to format CMIP emissions for OSCAR
def format_CMIP_emis(ds_in, 
    year_pi=1750, years_pi=(1750, 1759), years_pd=(2010, 2019), 
    scen_hist='hist', sect_endo=['bor', 'def', 'for', 'tem', 'gra'], keep_sect_NOx=False):

    ## endogenous/exogenous sectors    
    sect_exo = [sec for sec in ds_in.sect.values if sec not in sect_endo]

    ## list of species
    spc_list = list(set([var.split('_')[-1] for var in ds_in]))

    ## initialize as of PI (and drop biofuel data)
    For = ds_in.sel(year=slice(year_pi, None)).copy(deep=True)
    For = For.drop_vars([var for var in For if 'Ebf' in var])
    if 'scen' in ds_in.dims: For_hist = For.sel(scen=scen_hist, drop=True).dropna('year', how='all')
    else: For_hist = For.dropna('year', how='all')


    ## PI emissions
    ## anthropogenic
    for spc in [spc for spc in spc_list if f'Eant_{spc}' in For]:
        For[f'Eant_{spc}_pi'] = For_hist[f'Eant_{spc}'].sel(year=year_pi, drop=True).sum('sect', min_count=1)
        For[f'Eant_{spc}_pi'].attrs['units'] = For[f'Eant_{spc}'].units
        For[f'Eant_{spc}_pi'].attrs['year'] = year_pi
    ## split sectors for NOx (kept here as legacy)
    if keep_sect_NOx and 'Eant_NOx' in For:
        for sec in ['air', 'shp']:
            For[f'E{sec}_NOx_pi'] = For_hist['Eant_NOx'].sel(year=year_pi, sect=sec, drop=True)
            For[f'E{sec}_NOx_pi'].attrs['units'] = For['Eant_NOx'].units
            For[f'E{sec}_NOx_pi'].attrs['year'] = year_pi
    ## biomass burning (all)
    for spc in [spc for spc in spc_list if f'Ebb_{spc}' in For]:
        For[f'Ebb_{spc}_pi'] = For_hist[f'Ebb_{spc}'].sel(year=slice(*years_pi), drop=True).sum('sect', min_count=1).mean('year')
        For[f'Ebb_{spc}_pi'].attrs['units'] = For[f'Ebb_{spc}'].units
        For[f'Ebb_{spc}_pi'].attrs['years'] = years_pi
    ## biomass burning (excl. endo)
    for spc in [spc for spc in spc_list if f'Ebb_{spc}' in For]:
        For[f'Ebb2_{spc}_pi'] = For_hist[f'Ebb_{spc}'].sel(year=slice(*years_pi), drop=True).sel(sect=sect_exo).sum('sect', min_count=1).mean('year')
        For[f'Ebb2_{spc}_pi'].attrs['units'] = For[f'Ebb_{spc}'].units
        For[f'Ebb2_{spc}_pi'].attrs['years'] = years_pi


    ## PD emissions of precursors
    ## anthropogenic
    for spc in [spc for spc in spc_list if f'Eant_{spc}' in For]:
        For[f'Eant_{spc}_pd'] = For_hist[f'Eant_{spc}'].sel(year=slice(*years_pd), drop=True).sum('sect', min_count=1).mean('year')
        For[f'Eant_{spc}_pd'].attrs['units'] = For[f'Eant_{spc}'].units
        For[f'Eant_{spc}_pd'].attrs['years'] = years_pd
    ## split sectors for NOx (kept here as legacy)
    if keep_sect_NOx and 'Eant_NOx' in For:
        for sec in ['air', 'shp']:
            For[f'E{sec}_NOx_pd'] = For_hist['Eant_NOx'].sel(year=slice(*years_pd), sect=sec, drop=True).mean('year')
            For[f'E{sec}_NOx_pd'].attrs['units'] = For['Eant_NOx'].units
            For[f'E{sec}_NOx_pd'].attrs['years'] = years_pd
    ## biomass burning (all)
    for spc in [spc for spc in spc_list if f'Ebb_{spc}' in For]:
        For[f'Ebb_{spc}_pd'] = For_hist[f'Ebb_{spc}'].sel(year=slice(*years_pd), drop=True).sum('sect', min_count=1).mean('year')
        For[f'Ebb_{spc}_pd'].attrs['units'] = For[f'Ebb_{spc}'].units
        For[f'Ebb_{spc}_pd'].attrs['years'] = years_pd
    ## biomass burning (excl. endo)
    for spc in [spc for spc in spc_list if f'Ebb_{spc}' in For]:
        For[f'Ebb2_{spc}_pd'] = For_hist[f'Ebb_{spc}'].sel(year=slice(*years_pd), drop=True).sel(sect=sect_exo).sum('sect', min_count=1).mean('year')
        For[f'Ebb2_{spc}_pd'].attrs['units'] = For[f'Ebb_{spc}'].units
        For[f'Ebb2_{spc}_pd'].attrs['years'] = years_pd


    ## Delta forcings
    for var in [var for var in For if not var.endswith('_pi') and not var.endswith('_pd')]:
        if 'Eant_CO2' in var:
            For['D_Eff'] = For[var].sum('sect', min_count=1)
        elif 'Ebb_CO2' in var:
            pass
        elif 'Ebb_' in var:
            For['D_'+var] = For[var].sum('sect', min_count=1) - For[f'{var}_pi']
            For['D_'+var.replace('bb_', 'bb2_')] = For[var].sel(sect=sect_exo).sum('sect', min_count=1) - For[var.replace('bb_', 'bb2_')+'_pi']
        else:
            For['D_'+var] = For[var].sum('sect', min_count=1) - For[f'{var}_pi']
        if (var == 'Eant_NOx') & keep_sect_NOx:
            for sec in ['air', 'shp']:
                For[f'D_E{sec}_NOx'] = For_hist['Eant_NOx'].sel(sect=sec, drop=True)
                For[f'D_E{sec}_NOx'] -= For[f'E{sec}_NOx_pi']
                For[f'D_E{sec}_NOx_pd'].attrs['units'] = For['Eant_NOx'].units
        For = For.drop_vars(var)


    ## return 
    return For

