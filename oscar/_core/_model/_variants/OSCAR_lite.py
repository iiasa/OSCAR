import numpy as np
import xarray as xr


##################################################
##   OSCAR (lite version)
##################################################

## import and copy main model
from oscar._core._model.OSCAR import OSCAR
OSCAR_lite = OSCAR.copy(new_name='OSCAR_lite')


##==========
## Remove H2
##==========

## DROP

## H2 budget
from oscar._core._model._modules.OSCAR_H2 import OSCAR_H2
for proc in OSCAR_H2.proc_all:
    del OSCAR_lite[proc]

## RF contributions
del OSCAR_lite['ERF_swv_H2']
del OSCAR_lite['SARF_O3_H2']
del OSCAR_lite['ERF_ari_H2']


## REPLACE

## total stratospheric water vapor radiative forcing
OSCAR_lite.process(
    Out = 'ERF_swv', 
    In = ('ERF_swv_CH4',), 
    Eq = lambda Var, Par: Eq__ERF_swv(Var, Par), 
    units = 'W m-2')

def Eq__ERF_swv(Var, Par):
    return Var.ERF_swv_CH4


## total O3 radiative forcing (stratospheric-adjusted)
OSCAR_lite.process(
    Out = 'SARF_O3', 
    In = ('SARF_O3_Tg', 'SARF_O3_CH4', 'SARF_O3_N2O', 'SARF_O3_EESC', 'SARF_O3_CO', 'SARF_O3_VOC', 'SARF_O3_NOx', 'SARF_O3_BVOC', 'SARF_O3_LNOx'), 
    Eq = lambda Var, Par: Eq__SARF_O3(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3(Var, Par):
    return Var.SARF_O3_Tg + Var.SARF_O3_CH4 + Var.SARF_O3_N2O + Var.SARF_O3_EESC + Var.SARF_O3_CO + Var.SARF_O3_VOC + Var.SARF_O3_NOx + Var.SARF_O3_LNOx + Var.SARF_O3_BVOC


## aerosol-radiation interactions (aggregated)
OSCAR_lite.process(
    Out = 'ERF_ari', 
    In = ('ERF_ari_BC', 'ERF_ari_OC', 'ERF_ari_SO2', 'ERF_ari_NH3', 'ERF_ari_NOx', 'ERF_ari_VOC', 'ERF_ari_CH4', 'ERF_ari_N2O', 'ERF_ari_EESC', 'ERF_ari_dust', 'ERF_ari_salt', 'ERF_ari_DMS', 'ERF_ari_BVOC', 'ERF_ari_LNOx'), 
    Eq = lambda Var, Par: Eq__ERF_ari(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari(Var, Par):
    return Var.ERF_ari_BC + Var.ERF_ari_OC + Var.ERF_ari_SO2 + Var.ERF_ari_NH3 + Var.ERF_ari_NOx + Var.ERF_ari_VOC + Var.ERF_ari_CH4 + Var.ERF_ari_N2O + Var.ERF_ari_EESC + Var.ERF_ari_dust + Var.ERF_ari_salt + Var.ERF_ari_DMS + Var.ERF_ari_BVOC + Var.ERF_ari_LNOx


##==================
## Remove non-CO2 BB
##==================

## DROP

## non-CO2 BB emissions
from oscar._core._model._modules.OSCAR_fire import OSCAR_fire
for proc in OSCAR_fire.proc_all:
    del OSCAR_lite[proc]


## REPLACE

## net flux of geological CH4 oxidized in CO2
OSCAR_lite.process(
    Out = 'D_Foxi_CH4', 
    In = ('D_Eant_CH4', 'D_Ewet_CH4', 'D_Fsink_CH4'), 
    In2 = ('p_Egeo_CH4',),
    Eq = lambda Var, Par: Eq__D_Foxi_CH4(Var, Par), 
    units='PgC yr-1')

def Eq__D_Foxi_CH4(Var, Par):
    p_Egeo_CH4 = Var.p_Egeo_CH4 if 'p_Egeo_CH4' in Var else 0.
    D_Ebio = (1 - p_Egeo_CH4) * sum_reg(Var.D_Eant_CH4) + Var.D_Ewet_CH4
    D_Foxi = Var.D_Fsink_CH4
    return 1/Par.Pg_to_Tg * (D_Foxi - D_Ebio)


## atmospheric CH4 budget
OSCAR_lite.process(
    Out = 'd_CH4', 
    In = ('D_Eant_CH4', 'D_Ewet_CH4', 'D_Epf_CH4', 'D_Fsink_CH4', 'D_Emiss_CH4'), 
    Eq = lambda Var, Par: Eq__d_CH4(Var, Par),
    units = 'ppb yr-1')

def Eq__d_CH4(Var, Par):
    return 1 / Par.a_CH4 * (sum_reg(Var.D_Eant_CH4) + Var.D_Ewet_CH4 + Var.D_Epf_CH4 - Var.D_Fsink_CH4 + Var.D_Emiss_CH4)


## CH4 residual emissions (= budget imbalance)
OSCAR_lite.process(
    Out = 'Eimb_CH4', 
    In = ('d_CH4', 'D_Eant_CH4', 'D_Ewet_CH4', 'D_Epf_CH4', 'D_Fsink_CH4', 'D_Emiss_CH4'), 
    Eq = lambda Var, Par: Eq__Eimb_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__Eimb_CH4(Var, Par):
    return Par.a_CH4 * Var.d_CH4 - (sum_reg(Var.D_Eant_CH4) + Var.D_Ewet_CH4 + Var.D_Epf_CH4 - Var.D_Fsink_CH4 + Var.D_Emiss_CH4)


## atmospheric N2O budget
OSCAR_lite.process(
    Out = 'd_N2O', 
    In = ('D_Eant_N2O', 'D_Fsink_N2O', 'D_Emiss_N2O'), 
    Eq = lambda Var, Par: Eq__d_N2O(Var, Par),
    units = 'ppb yr-1')

def Eq__d_N2O(Var, Par):
    return 1 / Par.a_N2O * (sum_reg(Var.D_Eant_N2O) - Var.D_Fsink_N2O + Var.D_Emiss_N2O)


## N2O residual emissions (= budget imbalance)
OSCAR_lite.process(
    Out = 'Eimb_N2O', 
    In = ('d_N2O', 'D_Eant_N2O', 'D_Fsink_N2O', 'D_Emiss_N2O'), 
    Eq = lambda Var, Par: Eq__Eimb_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__Eimb_N2O(Var, Par):
    return Par.a_N2O * Var.d_N2O - (sum_reg(Var.D_Eant_N2O) - Var.D_Fsink_N2O + Var.D_Emiss_N2O)


## total preindustrial SO2 emissions
OSCAR_lite.process(
    Out = 'Etot_SO2_pi', 
    Eq = lambda Par: Eq__Etot_SO2_pi(Par), 
    units = 'TgS yr-1')

def Eq__Etot_SO2_pi(Par):
    return sum_reg(Par.Eant_SO2_pi)


## adjusted aerosol-radiation interaction efficiency to SO2 emissions
OSCAR_lite.process(
    Out = 'ph2_ari_SO2', 
    Eq = lambda Par: Eq__ph2_ari_SO2(Par), 
    units = 'W yr m-2 TgS-1')

def Eq__ph2_ari_SO2(Par):
    if 'Etot_SO2_pi' not in Par: return None
    Etot_SO2_pd = sum_reg(Par.Eant_SO2_pd)
    f_T = safe_exp(Par.g_tau_SO4 * Par.D_Tg_pd, 5.)
    return Par.ph_ari_SO2 * (Etot_SO2_pd - Par.Etot_SO2_pi) / (f_T * Etot_SO2_pd - Par.Etot_SO2_pi)


## aerosol-cloud interaction SO2 saturation emissions
OSCAR_lite.process(
    Out = 'Eaci_SO2', 
    Eq = lambda Par: Eq__Eaci_SO2(Par), 
    units = 'TgS yr-1')

def Eq__Eaci_SO2(Par):
    f_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi))
    f_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi))
    f_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi))
    return (1 + f_SO2 + f_BC + f_OC) / Par.i_aci_SO2


## aerosol-cloud interaction BC saturation emissions
OSCAR_lite.process(
    Out = 'Eaci_BC', 
    Eq = lambda Par: Eq__Eaci_BC(Par), 
    units = 'TgC yr-1')

def Eq__Eaci_BC(Par):
    f_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi))
    f_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi))
    f_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi))
    return (1 + f_SO2 + f_BC + f_OC) / Par.i_aci_BC


## aerosol-cloud interaction OC saturation emissions
OSCAR_lite.process(
    Out = 'Eaci_OC', 
    Eq = lambda Par: Eq__Eaci_OC(Par), 
    units = 'TgC yr-1')

def Eq__Eaci_OC(Par):
    f_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi))
    f_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi))
    f_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi))
    return (1 + f_SO2 + f_BC + f_OC) / Par.i_aci_OC


## O3 from CO
OSCAR_lite.process(
    Out = 'SARF_O3_CO', 
    In = ('D_Eant_CO',), 
    Eq = lambda Var, Par: Eq__SARF_O3_CO(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_CO(Var, Par):
    return Par.ph_O3_CO * (sum_reg(Var.D_Eant_CO))


## O3 from VOC
OSCAR_lite.process(
    Out = 'SARF_O3_VOC', 
    In = ('D_Eant_VOC',), 
    Eq = lambda Var, Par: Eq__SARF_O3_VOC(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_VOC(Var, Par):
    return Par.ph_O3_VOC * (sum_reg(Var.D_Eant_VOC))


## O3 from NOx
OSCAR_lite.process(
    Out = 'SARF_O3_NOx', 
    In = ('D_Eant_NOx',), 
    Eq = lambda Var, Par: Eq__SARF_O3_NOx(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_NOx(Var, Par):
    return Par.ph_O3_NOx * (sum_reg(Var.D_Eant_NOx))


## BC-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_BC', 
    In = ('D_Eant_BC',), 
    Eq = lambda Var, Par: Eq__ERF_ari_BC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_BC(Var, Par):
    return Par.ph_ari_BC * (sum_reg(Var.D_Eant_BC))


## OC-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_OC', 
    In = ('D_Eant_OC',), 
    Eq = lambda Var, Par: Eq__ERF_ari_OC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_OC(Var, Par):
    return Par.ph_ari_OC * (sum_reg(Var.D_Eant_OC))


## SO2-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_SO2', 
    In = ('D_Eant_SO2', 'D_Tg'), 
    Eq = lambda Var, Par: Eq__ERF_ari_SO2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_SO2(Var, Par):
    f_T = safe_exp(Par.g_tau_SO4 * Var.D_Tg, 5.)
    return Par.ph2_ari_SO2 * (f_T * (Par.Etot_SO2_pi + sum_reg(Var.D_Eant_SO2)) - Par.Etot_SO2_pi)


## NH3-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_NH3', 
    In = ('D_Eant_NH3',), 
    Eq = lambda Var, Par: Eq__ERF_ari_NH3(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_NH3(Var, Par):
    return Par.ph_ari_NH3 * (sum_reg(Var.D_Eant_NH3))


## NOx-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_NOx', 
    In = ('D_Eant_NOx',), 
    Eq = lambda Var, Par: Eq__ERF_ari_NOx(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_NOx(Var, Par):
    return Par.ph_ari_NOx * (sum_reg(Var.D_Eant_NOx))


## VOC-induced aerosol-radiation interactions
OSCAR_lite.process(
    Out = 'ERF_ari_VOC', 
    In = ('D_Eant_VOC',), 
    Eq = lambda Var, Par: Eq__ERF_ari_VOC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_VOC(Var, Par):
    return Par.ph_ari_VOC * (sum_reg(Var.D_Eant_VOC))


## aerosol-cloud interactions
OSCAR_lite.process(
    Out = 'ERF_aci', 
    In = ('D_Eant_SO2', 'D_Eant_OC', 'D_Eant_BC'), 
    Eq = lambda Var, Par: Eq__ERF_aci(Var, Par), 
    units = 'W m-2')

def Eq__ERF_aci(Var, Par):
    D_Eaci_SO2 = sum_reg(Var.D_Eant_SO2)
    D_Eaci_OC = sum_reg(Var.D_Eant_OC)
    D_Eaci_BC = sum_reg(Var.D_Eant_BC)
    return Par.Ph_aci * np.log1p(D_Eaci_SO2 / Par.Eaci_SO2 + D_Eaci_OC / Par.Eaci_OC  + D_Eaci_BC / Par.Eaci_BC)


## light-absorbing particles deposition on snow
OSCAR_lite.process(
    Out = 'ERF_lap', 
    In = ('D_Eant_BC',),
    Eq = lambda Var, Par: Eq__ERF_lap(Var, Par), 
    units = 'W m-2')

def Eq__ERF_lap(Var, Par):
    ## apply best model
    if 'reg_land' in Var.D_Eant_BC.dims: IRF_lap_ant = (Par.ph_lap_BC * Var.D_Eant_BC).sum('reg_land', min_count=1)
    else: IRF_lap_ant = Par.ph_lap_BC_glb * Var.D_Eant_BC
    return Par.a_adj_lap * (IRF_lap_ant)


##===========
## Remove LUC
##===========

## DROP

## all landC processes
from oscar._core._model._modules.OSCAR_landC import OSCAR_landC
for proc in OSCAR_landC.proc_all:
    del OSCAR_lite[proc]

## land ERF processes
del OSCAR_lite['ERF_luc']
del OSCAR_lite['ERF_lcc']


## ADD

## only landC_density processes
from oscar._core._model._modules.OSCAR_landC_density import OSCAR_landC_density
for proc in OSCAR_landC_density.proc_all:
    OSCAR_lite[proc] = OSCAR_landC_density[proc]

## cumulative LUC emissions (for total land carbon accounting)
OSCAR_lite.process(
    Out = 'D_Cluc', 
    In = ('D_Eluc',), 
    Eq = lambda Var, Par: DiffEq__D_Cluc(Var, Par), 
    units = 'PgC')

def DiffEq__D_Cluc(Var, Par):
    return D_Eluc


## REPLACE

## land carbon sink
OSCAR_lite.process(
    Out = 'D_Fland', 
    In = ('D_nbp',), 
    Eq = lambda Var, Par: Eq__D_Fland(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fland(Var, Par):
    return (Var.D_nbp * Par.Aland_pd).sum('bio_land', min_count=1)


## total NPP
OSCAR_lite.process(
    Out = 'NPP', 
    In = ('D_npp',), 
    Eq = lambda Var, Par: Eq__NPP(Var, Par), 
    units = 'PgC yr-1')

def Eq__NPP(Var, Par):
    NPP_env = (Par.npp_pi + Var.D_npp) * Par.Aland_pd
    return NPP_env


## total apparent NPP (after extraction)
OSCAR_lite.process(
    Out = 'NPPa', 
    In = ('D_npp', 'D_echarv', 'D_egraz'), 
    Eq = lambda Var, Par: Eq__NPPa(Var, Par), 
    units = 'PgC yr-1')

def Eq__NPPa(Var, Par):
    NPP_env = (Par.npp_pi * (1 - Par.p_charv - Par.p_graz) + Var.D_npp - Var.D_echarv - Var.D_egraz) * Par.Aland_pd
    return NPP_env


## total fire emissions
OSCAR_lite.process(
    Out = 'Efire', 
    In = ('D_efire',), 
    Eq = lambda Var, Par: Eq__Efire(Var, Par), 
    units = 'PgC yr-1')

def Eq__Efire(Var, Par):
    Efire_env = (Par.v_fire * Par.cveg_pi + Var.D_efire) * Par.Aland_pd
    return Efire_env


## total heterotrophic respiration
OSCAR_lite.process(
    Out = 'Eresp', 
    In = ('D_ecwd', 'D_esoil'), 
    Eq = lambda Var, Par: Eq__Eresp(Var, Par), 
    units = 'PgC yr-1')

def Eq__Eresp(Var, Par):
    Eresp_env = (Par.p_cwd_resp * Par.v_cwd * Par.ccwd_pi + Var.D_ecwd + Par.v_resp * Par.csoil_pi + Var.D_esoil) * Par.Aland_pd
    return Eresp_env


## total vegetation carbon stock
OSCAR_lite.process(
    Out = 'Cveg', 
    In = ('D_cveg',), 
    Eq = lambda Var, Par: Eq__Cveg(Var, Par), 
    units = 'PgC')

def Eq__Cveg(Var, Par):
    Cveg_env = (Par.cveg_pi + Var.D_cveg) * Par.Aland_pd
    return Cveg_env


## total coarse woody debris carbon stock
OSCAR_lite.process(
    Out = 'Ccwd', 
    In = ('D_ccwd',), 
    Eq = lambda Var, Par: Eq__Ccwd(Var, Par), 
    units = 'PgC')

def Eq__Ccwd(Var, Par):
    Ccwd_env = (Par.ccwd_pi + Var.D_ccwd) * Par.Aland_pd
    return Ccwd_env


## total soil carbon stock
OSCAR_lite.process(
    Out = 'Csoil', 
    In = ('D_csoil',), 
    Eq = lambda Var, Par: Eq__Csoil(Var, Par), 
    units = 'PgC')

def Eq__Csoil(Var, Par):
    Csoil_env = (Par.csoil_pi + Var.D_csoil) * Par.Aland_pd
    return Csoil_env


## preindustrial BVOC emissions
OSCAR_lite.process(
    Out = 'Enat_BVOC_pi', 
    Eq = lambda Par: Eq__Enat_BVOC_pi(Par), 
    units = 'Tg yr-1')

def Eq__Enat_BVOC_pi(Par):
    f_lcc = 1.
    f_T = safe_exp(Par.g_BVOC * Par.D_Tg_pd, 5.)
    return Par.Enat_BVOC_pd / f_lcc / f_T


## biogenic VOC emissions
OSCAR_lite.process(
    Out = 'D_Enat_BVOC', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_Enat_BVOC(Var, Par), 
    units = 'Tg yr-1')

def Eq__D_Enat_BVOC(Var, Par):
    f_lcc = 1.
    f_T = safe_exp(Par.g_BVOC * Var.D_Tg, 5.)
    return Par.Enat_BVOC_pi * (f_lcc * f_T - 1)


## total land carbon stock (excl. permafrost!)
OSCAR_landC.process(
    Out = 'Cland', 
    In = ('Cveg', 'Ccwd', 'Csoil', 'D_Cluc'), 
    Eq = lambda Var, Par: Eq__Cland(Var, Par), 
    units = 'PgC')

def Eq__Cland(Var, Par):
    return Var.Cveg + Var.Ccwd + Var.Csoil + Var.D_Cluc


##=================
## Shared processes
##=================

## REPLACE

## present-day tropospheric hydroxyl sink intensity
OSCAR_lite.process(
    Out = 'f_kOH_pd', 
    Eq = lambda Par: Eq__f_kOH_pd(Par), 
    units = '1')

def Eq__f_kOH_pd(Par):
    if 'EESC_pd' not in Par: return None
    if 'EESC_pi' not in Par: return None
    if 'Enat_BVOC_pi' not in Par: return None
    if 'Enat_LNOx_pd' not in Par: return None
    ## environmental factors
    f_kOH_CH4 = Par.ch_OH_CH4 * np.log(Par.CH4_pd / Par.CH4_pi)
    f_kOH_N2O = Par.ch_OH_N2O * np.log(Par.N2O_pd / Par.N2O_pi)
    f_kOH_EESC = Par.ch_OH_EESC * np.log(Par.EESC_pd.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_kOH_Tg = Par.ch_OH_Tg * np.log1p(Par.D_Tg_pd / Par.Tg_pi)
    ## natural emissions
    f_kOH_BVOC = Par.ch_OH_BVOC * np.log(sum_reg(Par.Enat_BVOC_pd) / sum_reg(Par.Enat_BVOC_pi))
    f_kOH_LNOx = Par.ch_OH_LNOx * np.log(Par.Enat_LNOx_pd / Par.Enat_LNOx_pi)
    ## anthropogenic emissions
    f_kOH_NOx = Par.ch_OH_NOx * np.log((sum_reg(Par.Eant_NOx_pd)) / (sum_reg(Par.Eant_NOx_pi)))
    f_kOH_CO = Par.ch_OH_CO * np.log((sum_reg(Par.Eant_CO_pd)) / (sum_reg(Par.Eant_CO_pi)))
    f_kOH_VOC = Par.ch_OH_VOC * np.log((sum_reg(Par.Eant_VOC_pd)) / (sum_reg(Par.Eant_VOC_pi)))
    ## return
    return np.exp(f_kOH_CH4 + f_kOH_N2O + f_kOH_EESC + f_kOH_Tg + f_kOH_BVOC + f_kOH_LNOx + f_kOH_NOx + f_kOH_CO + f_kOH_VOC)


## tropospheric hydroxyl sink intensity
OSCAR_lite.process(
    Out = 'f_kOH', 
    In = ('D_CH4', 'D_N2O', 'D_EESC', 'D_Tg', 'D_Enat_BVOC', 'D_Enat_LNOx', 'D_Eant_NOx', 'D_Eant_CO', 'D_Eant_VOC'), 
    Eq = lambda Var, Par: Eq__f_kOH(Var, Par), 
    units = '1')

def Eq__f_kOH(Var, Par):
    ## environmental factors
    f_kOH_CH4 = Par.ch_OH_CH4 * np.log1p(Var.D_CH4 / Par.CH4_pi)
    f_kOH_N2O = Par.ch_OH_N2O * np.log1p(Var.D_N2O / Par.N2O_pi)
    f_kOH_EESC = Par.ch_OH_EESC * np.log1p(Var.D_EESC.sel(age_air='mid_lat', drop=True) / Par.EESC_pi.sel(age_air='mid_lat', drop=True))
    f_kOH_Tg = Par.ch_OH_Tg * np.log1p(Var.D_Tg / Par.Tg_pi)
    ## natural emissions
    f_kOH_BVOC = Par.ch_OH_BVOC * np.log1p(sum_reg(Var.D_Enat_BVOC) / sum_reg(Par.Enat_BVOC_pi))
    f_kOH_LNOx = Par.ch_OH_LNOx * np.log1p(Var.D_Enat_LNOx / Par.Enat_LNOx_pi)
    ## anthropogenic emissions
    f_kOH_NOx = Par.ch_OH_NOx * np.log1p((sum_reg(Var.D_Eant_NOx)) / (sum_reg(Par.Eant_NOx_pi)))
    f_kOH_CO = Par.ch_OH_CO * np.log1p((sum_reg(Var.D_Eant_CO)) / (sum_reg(Par.Eant_CO_pi)))
    f_kOH_VOC = Par.ch_OH_VOC * np.log1p((sum_reg(Var.D_Eant_VOC)) / (sum_reg(Par.Eant_VOC_pi)))
    ## return
    return np.exp(f_kOH_CH4 + f_kOH_N2O + f_kOH_EESC + f_kOH_H2 + f_kOH_Tg + f_kOH_BVOC + f_kOH_LNOx + f_kOH_NOx + f_kOH_CO + f_kOH_VOC)

