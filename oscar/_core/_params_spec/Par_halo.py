import numpy as np
import xarray as xr

from oscar._core._params_spec.Cst import Cst


##==================
##==================

## atmospheric parameters and lifetimes for halogenated compounds
## based on AR6 WG1 and WMO/UNEP OA 2022
def get_params(**useless):

    ## initialization
    Par = xr.Dataset()

    ## dimensions
    ## list of 49 species in AR6 + Halon-1202 (because often reported as ODS)
    Par.coords['spc_halo'] = [
        'NF3', 'SF6', 'SO2F2', 
        'CF4', 'C2F6', 'C3F8', 'c-C4F8', 'C4F10', 'n-C5F12', 'n-C6F14', 'i-C6F14', 'C7F16', 'C8F18', # PFCs
        'HFC-23', 'HFC-32', 'HFC-125', 'HFC-134a', 'HFC-143a', 'HFC-152a', 'HFC-227ea', 'HFC-236fa', 'HFC-245fa', 'HFC-365mfc', 'HFC-43-10mee', # HFCs
        'CFC-11', 'CFC-12', 'CFC-13', 'CFC-112', 'CFC-112a', 'CFC-113', 'CFC-113a', 'CFC-114', 'CFC-114a', 'CFC-115', # CFCs
        'HCFC-22', 'HCFC-31', 'HCFC-124', 'HCFC-133a', 'HCFC-141b', 'HCFC-142b', # HCFCs
        'CH3Cl', 'CH2Cl2', 'CHCl3', 'CCl4', 'CH3CCl3', # HCCs
        'CH3Br', 'Halon-1211', 'Halon-1202', 'Halon-1301', 'Halon-2402'] # Halons


    ## chemical composition
    ## hydrogen
    n_H = xr.DataArray([
        0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # PFCs
        1, 2, 1, 2, 3, 4, 1, 2, 3, 5, 2, # HFCs
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # CFCs
        1, 2, 1, 2, 3, 3, # HCFCs
        3, 2, 1, 0, 3, # HCCs
        3, 0, 0, 0, 0 # Halons
        ], dims='spc_halo', attrs={'units': '1'})
    ## carbon
    n_C = xr.DataArray([
        0, 0, 0, 
        1, 2, 3, 4, 4, 5, 6, 6, 7, 8, # PFCs
        1, 1, 2, 2, 2, 2, 3, 3, 3, 4, 5, # HFCs
        1, 1, 1, 2, 2, 2, 2, 2, 2, 2, # CFCs
        1, 1, 2, 2, 2, 2, # HCFCs
        1, 1, 1, 1, 2, # HCCs
        1, 1, 1, 1, 2 # Halons
        ], dims='spc_halo', attrs={'units': '1'})
    ## nitrogen
    n_N = xr.zeros_like(Par.spc_halo).astype(int)
    n_N.loc['NF3'] = 1
    n_N.attrs['units'] = '1'
    ## oxygen
    n_O = xr.zeros_like(Par.spc_halo).astype(int)
    n_O.loc['SO2F2'] = 2
    n_O.attrs['units'] = '1'
    ## fluorine
    n_F = xr.DataArray([
        3, 6, 2, 
        4, 6, 8, 8, 10, 12, 14, 14, 16, 18, # PFCs
        3, 2, 5, 4, 3, 2, 7, 6, 5, 5, 10, # HFCs
        1, 2, 3, 2, 2, 3, 3, 4, 4, 5, # CFCs
        2, 1, 4, 3, 1, 2, # HCFCs
        0, 0, 0, 0, 0, # HCCs
        0, 2, 2, 3, 4 # Halons
        ], dims='spc_halo', attrs={'units': '1'})
    ## sulfur
    n_S = xr.zeros_like(Par.spc_halo).astype(int)
    n_S.loc[['SF6', 'SO2F2']] = 1
    n_S.attrs['units'] = '1'
    ## chlorine
    Par['n_Cl'] = xr.DataArray([
        0, 0, 0, 
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # PFCs
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # HFCs
        3, 2, 1, 4, 4, 3, 3, 2, 2, 1, # CFCs
        1, 1, 1, 1, 2, 1, # HCFCs
        1, 2, 3, 4, 3, # HCCs
        0, 1, 0, 0, 0 # Halons
        ], dims='spc_halo', attrs={'units': '1'})
    ## bromine
    Par['n_Br'] = xr.zeros_like(Par.spc_halo).astype(int)
    Par['n_Br'].loc[['CH3Br', 'Halon-1211', 'Halon-1301']] = 1
    Par['n_Br'].loc[['Halon-1202', 'Halon-2402']] = 2
    Par['n_Br'].attrs['units'] = '1'


    ## atmospheric conversion factor
    m_Xhalo = n_H * Cst.m_H + n_C * Cst.m_C + n_N * Cst.m_N + n_O * Cst.m_O + n_F * Cst.m_F + n_S * Cst.m_S + Par.n_Cl * Cst.m_Cl + Par.n_Br * Cst.m_Br
    Par['a_Xhalo'] = xr.DataArray(Cst.M_atm * m_Xhalo / Cst.m_air / 1E18, attrs={'units': 'Gg ppt-1'})


    ## preindustrial concentrations of halogenated compounds
    ## (Dentener et al., 2021; https://doi.org/10.1017/9781009157896.017) (Tables AIII.1b-f)
    Par['Xhalo_pi'] = xr.zeros_like(Par.spc_halo).astype(float)
    Par['Xhalo_pi'].loc['CF4'] = 34.05
    Par['Xhalo_pi'].loc['CH3Cl'] = 457.
    Par['Xhalo_pi'].loc['CH2Cl2'] = 7.
    Par['Xhalo_pi'].loc['CHCl3'] = 4.8
    Par['Xhalo_pi'].loc['CCl4'] = 0.03
    Par['Xhalo_pi'].loc['CH3Br'] = 5.30
    Par['Xhalo_pi'].attrs['units'] = 'ppt'


    ## present-day lifetimes
    ## (WMO, 2022; https://ozone.unep.org/science/assessment/sap) (Table A-5)
    ## note: unclear present-day period

    ## total
    v_Xhalo_total = xr.DataArray(np.array([
        569, 0.5*(850+1280), 36, 
        50000, 10000, 2600, 3200, 2600, 4100, 3100, 3100, 3000, 3000, # PFCs
        228, 5.27, 30.7, 13.5, 51.8, 1.5, 35.8, 213, 7.74, 8.86, 17, # HFCs
        52, 102, 640, 63.6, 52, 93, 55, 189, 105, 540, # CFCs
        11.6, 1.29, 5.9, 4.48, 8.81, 17.1, # HCFCs
        0.9, 176/365.25, 178/365.25, 30, 5, # HCCs
        0.8, 16, 2.5, 72, 28 # Halons
        ])**-1, dims='spc_halo', attrs={'units': 'yr-1'})

    ## tropospheric OH
    Par['v_Xhalo_OH_pd'] = xr.DataArray(np.array([
        np.inf, np.inf, 300, 
        np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, # PFCs
        243, 5.47, 32.3, 14.1, 57.2, 1.55, 37.5, 253, 8.16, 9.3, 17.9, # HFCs
        np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, # CFCs
        13, 1.33, 6.28, 4.74, 10.7, 19.3, # HCFCs
        1.57, 181/365.25, 183/365.25, np.inf, 6.1, # HCCs
        1.8, 1.45E+04, 122, 2.10E+04, 2.10E+04 # Halons
        ])**-1, dims='spc_halo', attrs={'units': 'yr-1'})

    ## stratospheric
    Par['v_Xhalo_hv_pd'] = xr.DataArray(np.array([
        740, np.inf, 630, 
        50000, 10000, 2600, 3200, 2600, 4100, 3100, 3100, 3000, 3000, # PFCs
        3636, 146, 665, 313, 548, 44.3, 754, 136, 153.8, 188, 360, # HFCs
        55, 103, np.inf, 65.4, 53.8, 94.5, 57.5, 191, 106.7, 664, # CFCs
        120, 36.7, 105, 82.6, 49.4, 148, # HCFCs
        30.4, np.inf, np.inf, 44, 38, # HCCs
        26.3, 41, 36, 73.5, 41 # Halons
        ])**-1, dims='spc_halo', attrs={'units': 'yr-1'})

    ## other (ocean, soil, aerosol, cloud)
    ## note: as residual
    Par['v_Xhalo_other'] = np.maximum(v_Xhalo_total - Par.v_Xhalo_OH_pd - Par.v_Xhalo_hv_pd, 0.)
    Par['v_Xhalo_other'].attrs['units'] = 'yr-1'


    ## RETURN
    return Par


##==================
##==================

'''
## FYI: soil lifetime (Table A-2)
v_Xhalo_soil = xr.zeros_like(Par_halo.spc_halo).astype(float)
v_Xhalo_soil.loc['CH3Cl'] = 1/4.2
v_Xhalo_soil.loc['CCl4'] = 1/375.
v_Xhalo_soil.loc['CH3Br'] = 1/3.35
v_Xhalo_soil.attrs['units'] = 'yr-1'

## FYI: ocean lifetime (Table A-2)
v_Xhalo_ocean = xr.zeros_like(Par_halo.spc_halo).astype(float)
v_Xhalo_ocean.loc['SO2F2'] = 1/40.
v_Xhalo_ocean.loc['HFC-125'] = 1/10650.
v_Xhalo_ocean.loc['HFC-134a'] = 1/5909.
v_Xhalo_ocean.loc['HFC-152a'] = 1/1958.
v_Xhalo_ocean.loc['HCFC-22'] = 1/1174.
v_Xhalo_ocean.loc['HCFC-124'] = 1/1855.
v_Xhalo_ocean.loc['HCFC-141b'] = 1/9190.
v_Xhalo_ocean.loc['HCFC-142b'] = 1/122200.
v_Xhalo_ocean.loc['CH3Cl'] = 1/12.
v_Xhalo_ocean.loc['CCl4'] = 1/124.
v_Xhalo_ocean.loc['CH3CCl3'] = 1/94.
v_Xhalo_ocean.loc['CH3Br'] = 1/3.1
v_Xhalo_ocean.attrs['units'] = 'yr-1'
'''

