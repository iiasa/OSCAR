import numpy as np
import xarray as xr

from scipy.stats import norm


##==================
##==================

## various constant values
def get_params(**useless):

    ## initialization
    Cst = xr.Dataset()


    ## Earth characteristics
    ## mass of the atmosphere
    ## (Trenberth & Smith, 2005; https://doi.org/10.1175/JCLI-3299.1) (Abstract)
    ## note: GCB uses a conversion factor for Catm of 2.124 which implies M_atm = 2.124 * 28.97 / 12.011 = 5.1230E18
    Cst['M_atm'] = xr.DataArray(5.1352E18, attrs={'units': 'kg'})

    ## Earth area
    ## note: can be derived from idealized sphere with mean radius of 6371 km
    ## source: https://nssdc.gsfc.nasa.gov/planetary/factsheet/earthfact.html
    Cst['A_Earth'] = xr.DataArray(510.1E12, attrs={'units': 'm2'})

    ## global ocean
    ## source: https://rwu.pressbooks.pub/webboceanography/chapter/1-1-overview-of-the-oceans/
    Cst['A_ocean'] = xr.DataArray(361.E12, attrs={'units': 'm2'})
    Cst['V_ocean'] = xr.DataArray(1.37E18, attrs={'units': 'm3'})


    ## molecular weights
    ## source: https://en.wikipedia.org/wiki/Standard_atomic_weight#List_of_atomic_weights
    ## note: using average isotopic composition ignores isotopic imbalance of processes
    ## elements
    Cst['m_H'] = xr.DataArray(1.0080, attrs={'units': 'g mol-1'})
    Cst['m_He'] = xr.DataArray(4.0026, attrs={'units': 'g mol-1'})
    Cst['m_C'] = xr.DataArray(12.011, attrs={'units': 'g mol-1'})
    Cst['m_N'] = xr.DataArray(14.007, attrs={'units': 'g mol-1'})
    Cst['m_O'] = xr.DataArray(15.999, attrs={'units': 'g mol-1'})
    Cst['m_F'] = xr.DataArray(18.998, attrs={'units': 'g mol-1'})
    Cst['m_Ne'] = xr.DataArray(20.180, attrs={'units': 'g mol-1'})
    Cst['m_P'] = xr.DataArray(30.974, attrs={'units': 'g mol-1'})
    Cst['m_S'] = xr.DataArray(32.06, attrs={'units': 'g mol-1'})
    Cst['m_Cl'] = xr.DataArray(35.45, attrs={'units': 'g mol-1'})
    Cst['m_Ar'] = xr.DataArray(39.95, attrs={'units': 'g mol-1'})
    Cst['m_Br'] = xr.DataArray(79.904, attrs={'units': 'g mol-1'})
    Cst['m_Kr'] = xr.DataArray(83.798, attrs={'units': 'g mol-1'})
    Cst['m_I'] = xr.DataArray(126.90, attrs={'units': 'g mol-1'})

    ## compounds
    Cst['m_H2'] = xr.DataArray(2*Cst.m_H, attrs={'units': 'g mol-1'})
    Cst['m_CO'] = xr.DataArray(Cst.m_C + Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_CO2'] = xr.DataArray(Cst.m_C + 2*Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_CH4'] = xr.DataArray(Cst.m_C + 4*Cst.m_H, attrs={'units': 'g mol-1'})
    Cst['m_N2'] = xr.DataArray(2*Cst.m_N, attrs={'units': 'g mol-1'})
    Cst['m_N2O'] = xr.DataArray(2*Cst.m_N + Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_NO'] = xr.DataArray(Cst.m_N + Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_NO2'] = xr.DataArray(Cst.m_N + 2*Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_NH3'] = xr.DataArray(Cst.m_N + 3*Cst.m_H, attrs={'units': 'g mol-1'})
    Cst['m_O2'] = xr.DataArray(2*Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_O3'] = xr.DataArray(3*Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_SO2'] = xr.DataArray(Cst.m_S + 2*Cst.m_O, attrs={'units': 'g mol-1'})
    Cst['m_DMS'] = xr.DataArray(Cst.m_S + 2*Cst.m_C + 6*Cst.m_H, attrs={'units': 'g mol-1'})
    Cst['m_isop'] = xr.DataArray(5*Cst.m_C + 8*Cst.m_H, attrs={'units': 'g mol-1'})

    ## dry air
    ## note: derived from present-day average composition, notably varies with CO2 and CH4 concentrations
    ## source: https://en.wikipedia.org/wiki/Atmosphere_of_Earth#Composition
    ## check: (78.08*Cst.m_N2 + 20.95*Cst.m_O2 + 0.9340*Cst.m_Ar + 0.0412*Cst.m_CO2 + 0.00182*Cst.m_Ne + 0.000524*Cst.m_He + 0.000179*Cst.m_CH4 + 0.000114*Cst.m_Kr) / 100
    Cst['m_air'] = xr.DataArray(28.97, attrs={'units': 'g mol-1'})


    ## conversion factors
    ## scales
    Cst['Pg_to_Tg'] = 1E3
    Cst['Tg_to_kg'] = 1E9
    Cst['kg_to_g'] = 1E3
    Cst['Mha_to_km2'] = 1E4
    Cst['km2_to_m2'] = 1E6
    ## scales
    Cst['yr_to_s'] = 3600 * 24 * 365.25 # Julian year
    Cst['yr_to_mon'] = 12
    ## uncertainty ranges
    Cst['s1_to_p90'] = norm.ppf(0.5+0.90/2)
    Cst['s1_to_p95'] = norm.ppf(0.5+0.95/2)
    Cst['s1_to_p99'] = norm.ppf(0.5+0.99/2)
    ## others
    Cst['degC_to_K'] = 273.15 # offset!
    Cst['Wyr_to_ZJ'] = Cst.yr_to_s / 1E21
    Cst['DU_to_TgO3'] = 2.687E16 * 1E4 * Cst.A_Earth * (6.02214076E23)**-1 * Cst.m_O3 * 1E-12 # [molecule cm-2] * [cm2 m-2] * [m2] * [mol molecule-1] * [g mol-1] * [g Tg-1]


    ## RETURN
    return Cst


##==================
##==================

## get params
Cst = get_params()

