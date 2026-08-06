import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_regions import aggreg_regions


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
            For['D_Eff'].attrs['units'] = For[var].units
        elif 'Ebb_CO2' in var:
            pass
        elif 'Ebb_' in var:
            For['D_'+var] = For[var].sum('sect', min_count=1) - For[f'{var}_pi']
            For['D_'+var.replace('bb_', 'bb2_')] = For[var].sel(sect=sect_exo).sum('sect', min_count=1) - For[var.replace('bb_', 'bb2_')+'_pi']
            For['D_'+var].attrs['units'] = For['D_'+var.replace('bb_', 'bb2_')].attrs['units'] = For[var].units
        else:
            For['D_'+var] = For[var].sum('sect', min_count=1) - For[f'{var}_pi']
            For['D_'+var].attrs['units'] = For[var].units
        if (var == 'Eant_NOx') & keep_sect_NOx:
            for sec in ['air', 'shp']:
                For[f'D_E{sec}_NOx'] = For_hist['Eant_NOx'].sel(sect=sec, drop=True)
                For[f'D_E{sec}_NOx'] -= For[f'E{sec}_NOx_pi']
                For[f'D_E{sec}_NOx'].attrs['units'] = For['Eant_NOx'].units
        For = For.drop_vars(var)


    ## return 
    return For


## function to format CMIP concentrations for OSCAR
def format_CMIP_conc(ds_in, conc_pi=None, t_lag=None, 
    year_pi=1750, years_pd=(2010, 2019), 
    scen_hist='hist'):

    ## list of species
    spc_list = ['CO2', 'CH4', 'N2O', 'H2', 'Xhalo']

    ## get default input if not provided
    if t_lag is None: 
        t_lag = xr.DataArray([3., 5.5], coords={'age_air': ['mid_lat', 'high_lat']}, dims=['age_air'], attrs={'units': 'yr'})

    ## initialize as of PI
    For = ds_in.sel(year=slice(year_pi, None)).copy(deep=True)
    if 'scen' in ds_in.dims: For_hist = For.sel(scen=scen_hist, drop=True).dropna('year', how='all')
    else: For_hist = For.dropna('year', how='all')


    ## PI concentrations
    ## note: only if not provided
    for spc in [spc for spc in spc_list if spc in For]:
        if conc_pi is None or (spc not in conc_pi and spc+'_pi' not in conc_pi):
            For[spc+'_pi'] = For_hist[spc].sel(year=year_pi, drop=True)
        elif spc in conc_pi:        
            For[spc+'_pi'] = conc_pi[spc]
        elif spc+'_pi' in conc_pi:  
            For[spc+'_pi'] = conc_pi[spc+'_pi']


    ## PD concentrations
    ## CH4, N2O, halogens
    if type(years_pd) == dict: years_dict = years_pd
    else: years_dict = {'CH4': years_pd, 'N2O': years_pd, 'Xhalo': years_pd}
    for spc in [spc for spc in spc_list if spc in For and spc in years_dict]:
        For[f'{spc}_pd'] = For_hist[spc].sel(year=slice(*years_dict[spc])).mean('year')
        For[f'{spc}_pd'].attrs['units'] = For_hist[spc].units
        For[f'{spc}_pd'].attrs['years'] = years_dict[spc]
    ## lagged halogens
    if 'Xhalo' in For and 'Xhalo' in years_dict:
        For['Xhalo_lag_pd'] = np.nan + xr.zeros_like(For.spc_halo).astype(float) + xr.zeros_like(t_lag)
        for age_air in t_lag.age_air.values:
            if float.is_integer(float(t_lag.loc[age_air])):
                For['Xhalo_lag_pd'].loc[:, age_air] = For_hist['Xhalo'].sel(year=slice(*[yr - int(t_lag.loc[age_air]) for yr in years_dict['Xhalo']])).mean('year')
            elif float.is_integer(float(t_lag.loc[age_air] - 0.5)): 
                For['Xhalo_lag_pd'].loc[:, age_air] = For_hist.Xhalo.rolling(year=2).mean().sel(year=slice(*[yr - int(t_lag.loc[age_air]) for yr in years_dict['Xhalo']])).mean('year')
        For['Xhalo_lag_pd'].attrs['units'] = For_hist[spc].units
        For['Xhalo_lag_pd'].attrs['years'] = years_dict[spc]


    ## Delta forcings and derivative
    for var in [var for var in For if not var.endswith('_pi') and not var.endswith('_pd')]:
        For['D_'+var] = For[var] - For[f'{var}_pi']
        For['D_'+var].attrs['units'] = For[var].units
        #For = For.drop_vars(var)

    ## derivatives
    for var in [var for var in For if var.startswith('D_')]:
        ## init
        For[var.replace('D_', 'd_')] = np.nan * For[var]
        ## historical
        For[var.replace('D_', 'd_')].loc[{'scen': scen_hist}] = For[var].mean('scen').differentiate('year').where(For[var].sel(scen=scen_hist).notnull())
        ## scenarios
        scenarios = [scen for scen in For.scen.values if scen != scen_hist]
        For[var.replace('D_', 'd_')].loc[{'scen': scenarios}] = For[var].sel(scen=scenarios).combine_first(For[var].sel(scen=scen_hist, drop=True)).differentiate('year').where(For[var].sel(scen=scenarios).notnull())
        ## extra
        For[var.replace('D_', 'd_')].loc[{'year': year_pi}] = 0.
        For[var.replace('D_', 'd_')].attrs['units'] = For[var].units + ' yr-1'
        For = For.drop_vars(var.replace('D_', ''))

    ## return 
    return For


##################################################
##   2. WRAPPERS TO GET STANDARD FORCINGS
##################################################

## get CMIP6/7 drivers for historical and scenarios
def get_CMIP_drivers(mod_region, CMIP='CMIP7', year_end=2300):
    print(f'creating {CMIP} drivers')

    ## get proper CMIP data
    if CMIP == 'CMIP7':
        year_hist = 2023
        ds_conc = load_data('concentrations_CMIP7.')
        ds_emis = aggreg_regions(load_data('emissions_CMIP7.'), mod_region)
        ds_land = aggreg_regions(load_data('land-use_LUH3.'), mod_region)
        ds_forc = load_data('radiative-forcing_RCMIP3.')
    elif CMIP == 'CMIP6':
        year_hist = 2014
        ds_conc = load_data('concentrations_CMIP6.')
        ds_emis = aggreg_regions(load_data('emissions_CMIP6.'), mod_region)
        ds_land = aggreg_regions(load_data('land-use_LUH2.'), mod_region)
        ds_land.coords['scen'] = ['ssp534-over' if scen == 'ssp534' else scen for scen in ds_land.scen.values]
        ds_land = xr.concat([ds_land, ds_land.sel(scen='ssp370').assign_coords(scen='ssp370-lowNTCF')], dim='scen', join='outer')
        ds_forc = load_data('radiative-forcing_AR6.')
    elif False:
        year_hist = 2000
        ds_conc = load_data('concentrations_CMIP5.')
        ds_emis = aggreg_regions(load_data('emissions_CMIP5.'), mod_region)
        ds_land = aggreg_regions(load_data('land-use_LUH1.'), mod_region)
        ds_forc = load_data('radiative-forcing_AR5.')
    else:
        raise ValueError("CMIP options available are: CMIP7, CMIP6")

    ## helper to properly cut hist/scen
    def _format_years(ds, year_hist, scen_hist='hist'):
        for scen in [scen for scen in ds.scen.values if scen != scen_hist]:
            ds.loc[{'scen': scen}] = xr.concat([ds.sel(scen=scen_hist), ds.sel(scen=scen)], dim='scen').mean('scen')
        return ds.where(((ds.scen == scen_hist) & (ds.year <= year_hist)) | ((ds.scen != scen_hist) & (ds.year >= year_hist)))

    ## initialization
    For = xr.Dataset()


    ## CONCENTRATIONS (first: provides the time axis)
    ## format input
    ds_tmp = format_CMIP_conc(_format_years(ds_conc, year_hist), 
        year_pi=1750, years_pd=(2010, 2019), scen_hist='hist')
    ## assign
    for var in ds_tmp: For[var] = ds_tmp[var]   


    ## EMISSIONS
    ## format input
    ds_tmp = format_CMIP_emis(_format_years(ds_emis, year_hist), 
        year_pi=1750, years_pi=(1750, 1760), years_pd=(2010, 2019), scen_hist='hist', sect_endo=['bor', 'def', 'for', 'tem', 'gra'])
    ds_tmp = ds_tmp.drop_vars([var for var in ds_tmp if 'Ebb_' in var])
    ds_tmp = ds_tmp.reindex(year=For.year.values)
    ds_tmp = ds_tmp.interpolate_na('year')
    ## assign
    for var in ds_tmp: For[var] = ds_tmp[var]


    ## LAND USE
    ## format input
    ds_tmp = format_LUH(_format_years(ds_land, year_hist), 
        year_pi=1750, year_pd=2022, scen_hist='hist')
    ## assign
    for var in ds_tmp: For[var] = ds_tmp[var]
     

    ## RADIATIVE FORCING
    ## format input
    ds_tmp = _format_years(ds_forc, year_hist)
    ds_tmp = ds_tmp.drop_vars([var for var in ds_forc if var not in ['ERF_aic', 'ERF_irrig', 'ERF_solar', 'ERF_volc']])
    ## assign
    for var in ds_tmp: For[var] = ds_tmp[var]


    ## COMMON STEPS
    ## remove present-day parameters
    ## note: related to processes and don't want to override those provided in Par
    for var in [var for var in For if '_pd' in var]:
        del For[var]

    ## turn off H2 (for now)
    For['D_Eant_H2'] = xr.DataArray(0. * For.year, coords=For.year.coords, attrs={'units': 'TgH2 yr-1'})
    For['D_Ebb_H2'] = xr.DataArray(0. * For.year, coords=For.year.coords, attrs={'units': 'TgH2 yr-1'})
    For['Eant_H2_pi'] = xr.DataArray(0., attrs={'units': 'TgH2 yr-1'})
    For['Ebb_H2_pi'] = xr.DataArray(0., attrs={'units': 'TgH2 yr-1'})
    For['D_H2'] = xr.DataArray(0. * For.year, coords=For.year.coords, attrs={'units': 'ppb'})
    For['d_H2'] = xr.DataArray(0. * For.year, coords=For.year.coords, attrs={'units': 'ppb yr-1'})
    For['H2_pi'] = xr.DataArray(536.2, attrs={'units': 'ppb'})
    For['H2_pd'] = xr.DataArray(536.2, attrs={'units': 'ppb'}) # this one we want!


    ## SPECIFIC STEPS
    if CMIP == 'CMIP7':
        for var in ['D_CO2', 'D_CH4', 'D_N2O', 'D_Xhalo'] + ['d_CO2', 'd_CH4', 'd_N2O', 'd_Xhalo']: 
            For[var].loc[{'year': year_hist, 'scen': 'hist'}] = For[var].sel(year=year_hist).mean('scen') # identical in all scenarios
        For['D_Eant_Xhalo'] = xr.zeros_like(For.year, dtype=np.float32) + xr.zeros_like(For.spc_halo, dtype=np.float32)
        For['D_Eant_Xhalo'].attrs['units'] = 'Gg yr-1'
    elif CMIP == 'CMIP6':
        For['D_Ebb2_N2O'] = For['D_Ebb2_N2O'].combine_first(For['D_Ebb2_N2O'].sel(scen='hist', drop=True).sel(year=slice(2010, 2014)).mean('year').where(For['D_Eff'].notnull()))
        for var in ['D_Eant_CH4', 'D_Eant_N2O', 'ERF_irrig']: 
            For[var] = xr.zeros_like(For.year, dtype=np.float32)
        For['D_Eant_CH4'].attrs['units'] = 'TgC yr-1'
        For['D_Eant_N2O'].attrs['units'] = 'TgN yr-1'
        For['ERF_irrig'].attrs['units'] = 'W m-2'
        For['D_Eant_Xhalo'] = xr.zeros_like(For.year, dtype=np.float32) + xr.zeros_like(For.spc_halo, dtype=np.float32)
        For['D_Eant_Xhalo'].attrs['units'] = 'Gg yr-1'


    ## RETURN
    For = For.sel(year=slice(None, year_end))
    return For.astype(np.float32)

