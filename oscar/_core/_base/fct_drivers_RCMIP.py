import numpy as np
import xarray as xr

from oscar._core._base.fct_load import load_data
from oscar._core._base.fct_regions import aggreg_regions
from oscar._core._base.fct_drivers_CMIP import format_LUH


##################################################
##   1. DATASET SPECIFIC FORMATTING
##################################################

## function to format RCMIP emissions for OSCAR
def format_RCMIP_emis(ds_in, 
    year_pi=1750, years_pd=(2010, 2019), 
    scen_hist='hist-cmip7'):

    ## list of species
    spc_list = list(set([var.split('_')[-1] for var in ds_in if len(var.split('_')) > 1]))
    spc_prec = ['BC', 'CO', 'OC', 'SO2', 'VOC', 'NH3', 'NOx']

    ## initialize as of PI
    For = ds_in.sel(year=slice(year_pi, None)).copy(deep=True)
    if 'scen' in ds_in.dims: For_hist = For.sel(scen=scen_hist, drop=True).dropna('year', how='all')
    else: For_hist = For.dropna('year', how='all')


    ## PI emissions
    ## anthropogenic
    for spc in [spc for spc in spc_list if f'Etot_{spc}' in For or f'Eant_{spc}' in For]:
        For[f'Eant_{spc}_pi'] = For_hist[f'Etot_{spc}' if f'Etot_{spc}' in For_hist else f'Eant_{spc}'].sel(year=year_pi, drop=True)
        For[f'Eant_{spc}_pi'].attrs['units'] = For[f'Etot_{spc}' if f'Etot_{spc}' in For_hist else f'Eant_{spc}'].units
        For[f'Eant_{spc}_pi'].attrs['year'] = year_pi
    ## biomass burning (all)
    for spc in [spc for spc in spc_list if f'Etot_{spc}' in For]:
        For[f'Ebb_{spc}_pi'] = 0 * For_hist[f'Etot_{spc}'].sel(year=year_pi, drop=True)
        For[f'Ebb_{spc}_pi'].attrs['units'] = For[f'Etot_{spc}'].units
        For[f'Ebb_{spc}_pi'].attrs['year'] = year_pi


    ## PD emissions of precursors
    ## anthropogenic
    for spc in [spc for spc in spc_prec if f'Etot_{spc}' in For]:
        For[f'Eant_{spc}_pd'] = For_hist[f'Etot_{spc}'].sel(year=slice(*years_pd), drop=True).mean('year')
        For[f'Eant_{spc}_pd'].attrs['units'] = For[f'Etot_{spc}'].units
        For[f'Eant_{spc}_pd'].attrs['years'] = years_pd
    ## biomass burning (all)
    for spc in [spc for spc in spc_list if f'Etot_{spc}' in For]:
        For[f'Ebb_{spc}_pd'] = 0 * For_hist[f'Etot_{spc}'].sel(year=slice(*years_pd), drop=True).mean('year')
        For[f'Ebb_{spc}_pd'].attrs['units'] = For[f'Etot_{spc}'].units
        For[f'Ebb_{spc}_pd'].attrs['years'] = years_pd


    ## Delta forcings
    for var in [var for var in For if not var.endswith('_pi') and not var.endswith('_pd')]:
        if 'Etot' in var:
            For['D_'+var.replace('Etot', 'Eant')] = For[var] - For[var.replace('Etot', 'Eant')+'_pi']
            For['D_'+var.replace('Etot', 'Eant')].attrs['units'] = For[var].units
            For['D_'+var.replace('Etot', 'Ebb')] = 0 * For[var]
            For['D_'+var.replace('Etot', 'Ebb')].attrs['units'] = For[var].units
        elif 'Eant' in var:
            For['D_'+var] = For[var] - For[var+'_pi']
            For['D_'+var].attrs['units'] = For[var].units
        else:
            For['D_'+var] = For[var]
            For['D_'+var].attrs['units'] = For[var].units
        For = For.drop_vars(var)


    ## return 
    return For


## function to format CMIP concentrations for OSCAR
def format_RCMIP_conc(ds_in, conc_pi=None, t_lag=None, 
    year_pi=1750, years_pd=(2010, 2019), 
    scen_hist='hist-cmip7'):

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


    ## Delta forcings
    for var in [var for var in For if not var.endswith('_pi') and not var.endswith('_pd')]:
        For['D_'+var] = For[var] - For[f'{var}_pi']
        For['D_'+var].attrs['units'] = For[var].units
        For = For.drop_vars(var)

    ## Derivatives
    for var in [var for var in For if var.startswith('D_')]:
        ## init
        For[var.replace('D_', 'd_')] = np.nan * For[var]
        ## historical
        scen_rest = [scen for scen in For.scen.values]
        if 'hist-cmip7' in For.scen:
            scen_cmip = [scen for scen in For.scen.values if For.scen_group.sel(scen=scen).str.contains('cmip7') and ~For.scen_group.sel(scen=scen).str.contains('var')]
            For[var.replace('D_', 'd_')].loc[{'scen': 'hist-cmip7'}] = For[var].sel(scen=scen_cmip).mean('scen').differentiate('year').where(For[var].sel(scen='hist-cmip7').notnull())
            scen_rest.remove('hist-cmip7')
            if 'hist-ext' in For.scen:
                For[var.replace('D_', 'd_')].loc[{'scen': 'hist-ext'}] = For[var].sel(scen=scen_cmip).mean('scen').differentiate('year').where(For[var].sel(scen='hist-cmip7').notnull(), 0.)
                scen_rest.remove('hist-ext')
        if 'hist-cmip6' in For.scen:
            scen_cmip = [scen for scen in For.scen.values if For.scen_group.sel(scen=scen).str.contains('cmip6') and ~For.scen_group.sel(scen=scen).str.contains('var')]
            For[var.replace('D_', 'd_')].loc[{'scen': 'hist-cmip6'}] = For[var].sel(scen=scen_cmip).mean('scen').differentiate('year').where(For[var].sel(scen='hist-cmip6').notnull())
            scen_rest.remove('hist-cmip6')
        ## others (historical included in scenarios in RCMIP)
        For[var.replace('D_', 'd_')].loc[{'scen': scen_rest}] = For[var].sel(scen=scen_rest).differentiate('year')
        ## extra
        For[var.replace('D_', 'd_')].loc[{'year': year_pi}] = 0.
        For[var.replace('D_', 'd_')].attrs['units'] = For[var].units + ' yr-1'
        #For = For.drop_vars(var.replace('D_', ''))


    ## return 
    return For


##################################################
##   2. WRAPPERS TO GET STANDARD FORCINGS
##################################################

## get RCMIP3 drivers
def get_RCMIP3_drivers(mod_region, variant='luc', year_end=2500):
    print(f'creating RCMIP3 drivers')

    ## load RCMIP3 data
    ds_emis = load_data('emissions_RCMIP3.')
    ds_conc = load_data('concentrations_RCMIP3.')
    ds_forc = load_data('radiative-forcing_RCMIP3.')

    ## extend CMIP7 historical to 2023
    ds_conc.loc[{'year': 2023, 'scen': 'hist-cmip7'}] = ds_conc.sel(year=2023, scen=[scen for scen in ds_conc.scen.values if ds_conc.scen_group.sel(scen=scen) == 'cmip7']).mean('scen')

    ## apply processing
    For_emis = format_RCMIP_emis(ds_emis, year_pi=1750, years_pd=(2010, 2019), scen_hist='hist-cmip7')
    For_conc = format_RCMIP_conc(ds_conc, year_pi=1750, years_pd=(2010, 2019), scen_hist='hist-cmip7')
    For_forc = ds_forc.fillna(0.)


    ## LOAD LAND USE DATA
    ## CMIP7-LUH3 data
    ds_luh3 = format_LUH(aggreg_regions(load_data('land-use_LUH3.'), mod_region), year_pi=1750, year_pd=2022, scen_hist='hist')
    ds_luh3.coords['scen'] = [scen.replace('hist', 'hist-cmip7') for scen in ds_luh3.scen.values]
    ds_luh3 = ds_luh3.reindex(scen=['hist-cmip7', 'h', 'hl', 'l', 'ln', 'm', 'ml', 'vl'])
    for var in [var for var in ds_luh3 if 'year' in ds_luh3[var].dims]:
        ds_luh3[var].loc[{'year': ds_luh3[var].sel(scen='hist-cmip7').dropna('year', how='all').year.values}] = ds_luh3[var].sel(scen='hist-cmip7').dropna('year', how='all')
    
    ## CMIP6-LUH2 data
    ds_luh2 = format_LUH(aggreg_regions(load_data('land-use_LUH2.'), mod_region), year_pi=1750, year_pd=2022, scen_hist='hist')
    ds_luh2 = ds_luh2.sel(scen=[scen for scen in ds_luh2.scen.values if scen not in ['hist-high', 'hist-low']])
    ds_luh2.coords['scen'] = [scen.replace('ssp534', 'ssp534-over').replace('hist', 'hist-cmip6') for scen in ds_luh2.scen.values]
    for var in [var for var in ds_luh2 if 'year' in ds_luh2[var].dims]:
        ds_luh2[var].loc[{'year': ds_luh2[var].sel(scen='hist-cmip6').dropna('year', how='all').year.values}] = ds_luh2[var].sel(scen='hist-cmip6').dropna('year', how='all')

    ## merge land use data
    for var in [var for var in ds_luh3 if 'scen' not in ds_luh3[var].dims]:
        ds_luh3[var] = ds_luh3[var] + xr.zeros_like(ds_luh3.scen, dtype=float)
    for var in [var for var in ds_luh2 if 'scen' not in ds_luh2[var].dims]:
        ds_luh2[var] = ds_luh2[var] + xr.zeros_like(ds_luh2.scen, dtype=float)
    ds_land = xr.concat([ds_luh3, ds_luh2], join='outer', data_vars='all', dim='scen')
    ds_land['Aland_pd'] = ds_luh3['Aland_pd'].mean('scen')

    ## extend land use data
    ds_land = ds_land.reindex(year=np.arange(1750, 2500+1))
    for var in ds_land:
        if var in ['D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dC_wharv1']:
            ds_land[var] = ds_land[var].combine_first(0 * ds_land.year.sel(year=slice(None, 2299)) + 0 * ds_land[var].sel(year=2099, drop=True))
            ds_land[var] = ds_land[var].combine_first(0 * ds_land.year + 0 * ds_land[var].sel(year=2299, drop=True))
        elif 'year' in ds_land[var].dims:
            ds_land[var] = ds_land[var].combine_first(0 * ds_land.year.sel(year=slice(None, 2299)) + ds_land[var].sel(year=2099, drop=True))
            ds_land[var] = ds_land[var].combine_first(0 * ds_land.year + ds_land[var].sel(year=2299, drop=True))
    
    ## variant-specific land use data
    if variant == 'luc':
        For_land = ds_land.copy(deep=True)
        For_emis = For_emis.drop_vars('D_Eluc')
        For_forc = For_forc.drop_vars('ERF_luc')
    elif variant == 'lite':
        For_land = 0 * ds_land.fillna(0.)
        for var in [var for var in For_land if 'scen' in var]:
            For_land[var].loc[{'scen': 'hist-cmip7'}] = For_land[var].sel(scen='hist-cmip7').where(For_land.year <= 2023)
            For_land[var].loc[{'scen': 'hist-cmip6'}] = For_land[var].sel(scen='hist-cmip6').where(For_land.year <= 2014)
        For_land['Aland_pi'] = ds_land['Aland_pd']
        For_land['Aland_pd'] = ds_land['Aland_pd']
    else:
        raise ValueError("Only variants available are: luc, lite")


    ## GAP FILL SCENARIOS
    ## list scenarios by group
    scen_cmip7 = list(set([scen for scen in ds_emis.scen.values if ds_emis.scen_group.sel(scen=scen).str.contains('cmip7')]) 
        | set([scen for scen in ds_conc.scen.values if ds_conc.scen_group.sel(scen=scen).str.contains('cmip7')]) 
        | set([scen for scen in ds_forc.scen.values if ds_forc.scen_group.sel(scen=scen).str.contains('cmip7')]))
    scen_cmip6 = list(set([scen for scen in ds_emis.scen.values if ds_emis.scen_group.sel(scen=scen).str.contains('cmip6')]) 
        | set([scen for scen in ds_conc.scen.values if ds_conc.scen_group.sel(scen=scen).str.contains('cmip6')]) 
        | set([scen for scen in ds_forc.scen.values if ds_forc.scen_group.sel(scen=scen).str.contains('cmip6')]))
    scen_idealE = [scen for scen in ds_emis.scen.values if ds_emis.scen_group.sel(scen=scen).str.contains('ideal')]
    scen_idealC = [scen for scen in ds_conc.scen.values if ds_conc.scen_group.sel(scen=scen).str.contains('ideal')]

    ## expand scenario dim
    scen_list = scen_cmip7 + scen_cmip6 + [scen for scen in scen_idealE if scen not in ['piControl', 'hist-ext']] + scen_idealC
    For_emis = For_emis.reindex(scen=scen_list)
    For_conc = For_conc.reindex(scen=scen_list)
    For_forc = For_forc.reindex(scen=scen_list)
    For_land = For_land.reindex(scen=scen_list)

    ##------------
    ## CMIP7 emis.
    if len(set(scen_cmip7) - set(ds_emis.scen.values)) > 0:
        raise RuntimeError("No gap filling for emissions for: {}".format(set(scen_cmip7) - set(For_emis.scen.values)))
    
    ## CMIP7 conc.
    for scen in set(scen_cmip7) - set(ds_conc.scen.values):
        for var in [var for var in For_conc if 'scen' in For_conc[var].dims]:
            if '-CH4' in scen:
                if 'CH4' not in var: For_conc[var].loc[{'scen': scen}] = For_conc[var].sel(scen=scen.split('-')[0])
                else: For_conc[var].loc[{'scen': scen}] = For_conc[var].sel(scen=scen.split('-CH4')[-1])
            else:
                For_conc[var].loc[{'scen': scen}] = For_conc[var].sel(scen=scen.split('-')[0])

    ## CMIP7 forc.
    for scen in set(scen_cmip7) - set(ds_forc.scen.values):
        for var in [var for var in For_forc if 'scen' in For_forc[var].dims]:
            For_forc[var].loc[{'scen': scen}] = For_forc[var].sel(scen=scen.split('-')[0])

    ## CMIP7 land
    for scen in set(scen_cmip7) - set(ds_land.scen.values):
        for var in [var for var in For_land if 'scen' in For_land[var].dims]:
            For_land[var].loc[{'scen': scen}] = For_land[var].sel(scen=scen.split('-')[0])

    ##------------
    ## CMIP6 emis.
    if len(set(scen_cmip6) - set(ds_emis.scen.values)) > 0:
        raise RuntimeError("No gap filling for emissions for: {}".format(set(scen_cmip6) - set(For_emis.scen.values)))

    ## CMIP6 conc.
    for scen in set(scen_cmip6) - set(ds_conc.scen.values):
        for var in [var for var in For_conc if 'scen' in For_conc[var].dims]:
            if 'methanemip' in scen:            
                if 'CH4' not in var: For_conc[var].loc[{'scen': scen}] = For_conc[var].sel(scen='ssp245')
            else:
                For_forc[var].loc[{'scen': scen}] = For_forc[var].sel(scen=scen.split('-')[0] if '-over' not in scen else '-'.join(scen.split('-')[:-1]))

    ## CMIP6 forc.
    for scen in set(scen_cmip6) - set(ds_forc.scen.values):
        for var in [var for var in For_forc if 'scen' in For_forc[var].dims]:
            if 'methanemip' in scen:
                For_forc[var].loc[{'scen': scen}] = For_forc[var].sel(scen='ssp245')
            else:
                For_forc[var].loc[{'scen': scen}] = For_forc[var].sel(scen=scen.split('-')[0] if '-over' not in scen else '-'.join(scen.split('-')[:-1]))

    ## CMIP6 land
    for scen in set(scen_cmip6) - set(ds_land.scen.values):
        for var in [var for var in For_land if 'scen' in For_land[var].dims]:
            if 'methanemip' in scen:
                For_land[var].loc[{'scen': scen}] = For_land[var].sel(scen='ssp245')
            else:
                For_land[var].loc[{'scen': scen}] = For_land[var].sel(scen=scen.split('-')[0] if '-over' not in scen else '-'.join(scen.split('-')[:-1]))

    ##-------------
    ## idealE emis.
    if len(set(scen_idealE) - set(ds_emis.scen.values)) > 0:
        raise RuntimeError("No gap filling for emissions for: {}".format(set(scen_idealE) - set(For_emis.scen.values)))
    for scen in For_emis.scen.values:
        for var in [var for var in For_emis if 'scen' in For_emis[var].dims]:
            if For_emis[var].sel(scen=scen).isnull().all():
                For_emis[var].loc[{'scen': scen}] = For_emis[var].sel(scen='piControl')

    ## idealE conc.
    for scen in set(scen_idealE) - set(ds_conc.scen.values):
        for var in [var for var in For_conc if 'scen' in For_conc[var].dims]:
            if 'CO2' not in var: For_conc[var].loc[{'scen': scen}] = For_conc[var].sel(scen='piControl')

    ## idealE forc.
    for scen in set(scen_idealE) - set(ds_forc.scen.values):
        for var in [var for var in For_forc if 'scen' in For_forc[var].dims]:
            For_forc[var].loc[{'scen': scen}] = 0.

    ## idealE land
    for scen in set(scen_idealE) - set(ds_land.scen.values):
        for var in [var for var in For_land if 'scen' in For_land[var].dims]:
            if 'year' in For_land[var].dims:
                For_land[var].loc[{'scen': scen}] = 0.
            else:
                For_land[var].loc[{'scen': scen}] = For_land[var].sel(scen='hist-cmip7')
    
    ##-------------
    ## idealC emis.
    for scen in set(scen_idealC) - set(ds_emis.scen.values):
        for var in [var for var in For_emis if 'scen' in For_emis[var].dims]:
            For_emis[var].loc[{'scen': scen}] = For_emis[var].sel(scen='piControl')

    ## idealC conc.
    if len(set(scen_idealC) - set(ds_conc.scen.values)) > 0:
        raise RuntimeError("No gap filling for concentrations for: {}".format(set(scen_idealC) - set(For_conc.scen.values)))
    for scen in ['hist-ext'] if 'hist-ext' in scen_idealC else []:
        for var in [var for var in For_conc if 'scen' in For_conc[var].dims]:
            if var in ['d_CO2', 'd_CH4', 'd_N2O', 'd_Xhalo']:
                For_conc[var].loc[{'scen': scen, 'year': slice(2023, None)}] = 0.
    
    ## idealC forc.
    for scen in set(scen_idealC) - set(ds_forc.scen.values):
        for var in [var for var in For_forc if 'scen' in For_forc[var].dims]:
            if scen == 'hist-ext':
                For_forc[var].loc[{'scen': scen, 'year': slice(None, 2022)}] = For_forc[var].sel(year=slice(None, 2022), scen='hist-cmip7')
                For_forc[var].loc[{'scen': scen, 'year': slice(2023, None)}] = For_forc[var].sel(year=2022, scen='hist-cmip7')
            else:
                For_forc[var].loc[{'scen': scen}] = 0.

    ## idealC land
    for scen in set(scen_idealC) - set(ds_land.scen.values):
        for var in [var for var in For_land if 'scen' in For_land[var].dims]:
            if 'year' in For_land[var].dims:
                if scen == 'hist-ext':
                    For_land[var].loc[{'scen': scen, 'year': slice(None, 2022)}] = For_land[var].sel(year=slice(None, 2022), scen='hist-cmip7')
                    if var in ['D_dA_lcc1', 'D_dA_lcc2', 'D_dA_wharv1', 'D_dC_wharv1']:
                        For_land[var].loc[{'scen': scen, 'year': slice(2023, None)}] = 0.
                    else:
                        For_land[var].loc[{'scen': scen, 'year': slice(2023, None)}] = For_land[var].sel(year=2022, scen='hist-cmip7')
                else:
                    For_land[var].loc[{'scen': scen}] = 0.
            else:
                For_land[var].loc[{'scen': scen}] = For_land[var].sel(scen='hist-cmip7')


    ## FINALISATION
    ## merge forcings (and order scenarios)
    For = xr.merge([For_emis, For_conc, For_forc, For_land], join='outer', compat='no_conflicts')
    For = For.sel(scen=np.sort(For.scen.values))

    ## remove present-day parameters
    ## note: off for now -- this would use default parameters instead
    if False:
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


    ## RETURN
    For = For.sel(year=slice(None, year_end))
    return For.astype(np.float32)

