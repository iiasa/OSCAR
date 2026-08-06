import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg, safe_exp


#####################################################################
##   EFFECTIVE RADIATIVE FORCING - SHORT-LIVED CLIMATE FORCERS
#####################################################################

## initialize
OSCAR_ERF_slcf = Model('OSCAR_ERF_slcf')

## module adapted from FAIR:
## (Smith et al., 2024; https://doi.org/10.5194/gmd-17-8569-2024)


##=====================
## Secondary parameters
##=====================

## total preindustrial SO2 emissions
OSCAR_ERF_slcf.process(
    Out = 'Etot_SO2_pi', 
    Eq = lambda Par: Eq__Etot_SO2_pi(Par), 
    units = 'TgS yr-1')

def Eq__Etot_SO2_pi(Par):
    if 'Ebb_SO2_pi' not in Par: return None
    return sum_reg(Par.Eant_SO2_pi) + sum_reg(Par.Ebb_SO2_pi)


## adjusted aerosol-radiation interaction efficiency to SO2 emissions
## note: this is required because of the added climate effect on the sulfate lifetime
OSCAR_ERF_slcf.process(
    Out = 'ph2_ari_SO2', 
    Eq = lambda Par: Eq__ph2_ari_SO2(Par), 
    units = 'W yr m-2 TgS-1')

def Eq__ph2_ari_SO2(Par):
    if 'Etot_SO2_pi' not in Par: return None
    Etot_SO2_pd = sum_reg(Par.Eant_SO2_pd) + sum_reg(Par.Ebb_SO2_pd)
    fct_T = safe_exp(Par.g_tau_SO4 * Par.D_Tg_pd, 5.)
    return Par.ph_ari_SO2 * (Etot_SO2_pd - Par.Etot_SO2_pi) / (fct_T * Etot_SO2_pd - Par.Etot_SO2_pi)


## aerosol-radiation interaction efficiency to mineral dust emissions
OSCAR_ERF_slcf.process(
    Out = 'ph_ari_dust', 
    Eq = lambda Par: Eq__ph_ari_dust(Par), 
    units = 'W yr m-2 Tg-1')

def Eq__ph_ari_dust(Par):
    return Par.Ph_ari_dust / Par.Edust_pi


## aerosol-radiation interaction efficiency to sea salt emissions
OSCAR_ERF_slcf.process(
    Out = 'ph_ari_salt', 
    Eq = lambda Par: Eq__ph_ari_salt(Par), 
    units = 'W yr m-2 Tg-1')

def Eq__ph_ari_salt(Par):
    return Par.Ph_ari_salt / Par.Esalt_pi


## aerosol-cloud interaction SO2 saturation emissions
OSCAR_ERF_slcf.process(
    Out = 'Eaci_SO2', 
    Eq = lambda Par: Eq__Eaci_SO2(Par), 
    units = 'TgS yr-1')

def Eq__Eaci_SO2(Par):
    if 'Ebb_SO2_pi' not in Par: return None
    if 'Ebb_BC_pi' not in Par: return None
    if 'Ebb_OC_pi' not in Par: return None
    fct_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi) + sum_reg(Par.Ebb_SO2_pi))
    fct_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi) + sum_reg(Par.Ebb_BC_pi))
    fct_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi) + sum_reg(Par.Ebb_OC_pi))
    return (1 + fct_SO2 + fct_BC + fct_OC) / Par.i_aci_SO2


## aerosol-cloud interaction BC saturation emissions
OSCAR_ERF_slcf.process(
    Out = 'Eaci_BC', 
    Eq = lambda Par: Eq__Eaci_BC(Par), 
    units = 'TgC yr-1')

def Eq__Eaci_BC(Par):
    if 'Ebb_SO2_pi' not in Par: return None
    if 'Ebb_BC_pi' not in Par: return None
    if 'Ebb_OC_pi' not in Par: return None
    fct_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi) + sum_reg(Par.Ebb_SO2_pi))
    fct_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi) + sum_reg(Par.Ebb_BC_pi))
    fct_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi) + sum_reg(Par.Ebb_OC_pi))
    return (1 + fct_SO2 + fct_BC + fct_OC) / Par.i_aci_BC


## aerosol-cloud interaction OC saturation emissions
OSCAR_ERF_slcf.process(
    Out = 'Eaci_OC', 
    Eq = lambda Par: Eq__Eaci_OC(Par), 
    units = 'TgC yr-1')

def Eq__Eaci_OC(Par):
    if 'Ebb_SO2_pi' not in Par: return None
    if 'Ebb_BC_pi' not in Par: return None
    if 'Ebb_OC_pi' not in Par: return None
    fct_SO2 = Par.i_aci_SO2 * (sum_reg(Par.Eant_SO2_pi) + sum_reg(Par.Ebb_SO2_pi))
    fct_BC = Par.i_aci_BC * (sum_reg(Par.Eant_BC_pi) + sum_reg(Par.Ebb_BC_pi))
    fct_OC = Par.i_aci_OC * (sum_reg(Par.Eant_OC_pi) + sum_reg(Par.Ebb_OC_pi))
    return (1 + fct_SO2 + fct_BC + fct_OC) / Par.i_aci_OC


##=====================
## Diagnostic variables
##=====================

## stratospheric water vapor from CH4 oxidation
OSCAR_ERF_slcf.process(
    Out = 'ERF_swv_CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__ERF_swv_CH4(Var, Par), 
    units = 'W m-2')

def Eq__ERF_swv_CH4(Var, Par):
    return Par.Ph_swv_CH4 / Par.x_rf_swv * ((1 + Var.D_CH4 / Par.CH4_pi) ** Par.x_rf_swv - 1)


## stratospheric water vapor from H2 oxidation
OSCAR_ERF_slcf.process(
    Out = 'ERF_swv_H2', 
    In = ('D_H2',), 
    Eq = lambda Var, Par: Eq__ERF_swv_H2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_swv_H2(Var, Par):
    return Par.ph_swv_H2 * Var.D_H2


## total stratospheric water vapor radiative forcing
OSCAR_ERF_slcf.process(
    Out = 'ERF_swv', 
    In = ('ERF_swv_CH4', 'ERF_swv_H2'), 
    Eq = lambda Var, Par: Eq__ERF_swv(Var, Par), 
    units = 'W m-2')

def Eq__ERF_swv(Var, Par):
    return Var.ERF_swv_CH4 + Var.ERF_swv_H2


## O3 from climate change
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_Tg', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__SARF_O3_Tg(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_Tg(Var, Par):
    return Par.ph_O3_Tg * Var.D_Tg


## O3 from CH4
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__SARF_O3_CH4(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_CH4(Var, Par):
    return Par.Ph_O3_CH4 / Par.x_rf_O3 * ((1 + Var.D_CH4 / Par.CH4_pi) ** Par.x_rf_O3 - 1)


## O3 from N2O
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_N2O', 
    In = ('D_N2O',), 
    Eq = lambda Var, Par: Eq__SARF_O3_N2O(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_N2O(Var, Par):
    return Par.ph_O3_N2O * Var.D_N2O


## O3 from EESC
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_EESC', 
    In = ('D_EESC',), 
    Eq = lambda Var, Par: Eq__SARF_O3_EESC(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_EESC(Var, Par):
    return Par.ph_O3_EESC * Var.D_EESC.sel(age_air='mid_lat', drop=True)


## O3 from CO
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_CO', 
    In = ('D_Eant_CO', 'D_Ebb_CO'), 
    Eq = lambda Var, Par: Eq__SARF_O3_CO(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_CO(Var, Par):
    return Par.ph_O3_CO * (sum_reg(Var.D_Eant_CO) + sum_reg(Var.D_Ebb_CO))


## O3 from VOC
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_VOC', 
    In = ('D_Eant_VOC', 'D_Ebb_VOC'), 
    Eq = lambda Var, Par: Eq__SARF_O3_VOC(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_VOC(Var, Par):
    return Par.ph_O3_VOC * (sum_reg(Var.D_Eant_VOC) + sum_reg(Var.D_Ebb_VOC))


## O3 from NOx
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_NOx', 
    In = ('D_Eant_NOx', 'D_Ebb_NOx'), 
    Eq = lambda Var, Par: Eq__SARF_O3_NOx(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_NOx(Var, Par):
    return Par.ph_O3_NOx * (sum_reg(Var.D_Eant_NOx) + sum_reg(Var.D_Ebb_NOx))


## O3 from H2
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_H2', 
    In = ('D_H2',), 
    Eq = lambda Var, Par: Eq__SARF_O3_H2(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_H2(Var, Par):
    return Par.ph_O3_H2 * Var.D_H2


## O3 from BVOC
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_BVOC', 
    In = ('D_Enat_BVOC',), 
    Eq = lambda Var, Par: Eq__SARF_O3_BVOC(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_BVOC(Var, Par):
    return Par.ph_O3_BVOC * Var.D_Enat_BVOC


## O3 from LNOx
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3_LNOx', 
    In = ('D_Enat_LNOx',), 
    Eq = lambda Var, Par: Eq__SARF_O3_LNOx(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3_LNOx(Var, Par):
    return Par.ph_O3_LNOx * Var.D_Enat_LNOx


## total O3 radiative forcing (stratospheric-adjusted)
OSCAR_ERF_slcf.process(
    Out = 'SARF_O3', 
    In = ('SARF_O3_Tg', 'SARF_O3_CH4', 'SARF_O3_N2O', 'SARF_O3_EESC', 'SARF_O3_CO', 'SARF_O3_VOC', 'SARF_O3_NOx', 'SARF_O3_H2', 'SARF_O3_BVOC', 'SARF_O3_LNOx'), 
    Eq = lambda Var, Par: Eq__SARF_O3(Var, Par), 
    units = 'W m-2')

def Eq__SARF_O3(Var, Par):
    return Var.SARF_O3_Tg + Var.SARF_O3_CH4 + Var.SARF_O3_N2O + Var.SARF_O3_EESC + Var.SARF_O3_CO + Var.SARF_O3_VOC + Var.SARF_O3_NOx + Var.SARF_O3_H2 + Var.SARF_O3_LNOx + Var.SARF_O3_BVOC


## total O3 effective radiative forcing
OSCAR_ERF_slcf.process(
    Out = 'ERF_O3', 
    In = ('SARF_O3',), 
    Eq = lambda Var, Par: Eq__ERF_O3(Var, Par), 
    units = 'W m-2')

def Eq__ERF_O3(Var, Par):
    return Par.a_adj_O3 * Var.SARF_O3


## BC-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_BC', 
    In = ('D_Eant_BC', 'D_Ebb_BC'), 
    Eq = lambda Var, Par: Eq__ERF_ari_BC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_BC(Var, Par):
    return Par.ph_ari_BC * (sum_reg(Var.D_Eant_BC) + sum_reg(Var.D_Ebb_BC))


## OC-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_OC', 
    In = ('D_Eant_OC', 'D_Ebb_OC'), 
    Eq = lambda Var, Par: Eq__ERF_ari_OC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_OC(Var, Par):
    return Par.ph_ari_OC * (sum_reg(Var.D_Eant_OC) + sum_reg(Var.D_Ebb_OC))


## SO2-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_SO2', 
    In = ('D_Eant_SO2', 'D_Ebb_SO2', 'D_Tg'), 
    Eq = lambda Var, Par: Eq__ERF_ari_SO2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_SO2(Var, Par):
    fct_T = safe_exp(Par.g_tau_SO4 * Var.D_Tg, 5.)
    return Par.ph2_ari_SO2 * (fct_T * (Par.Etot_SO2_pi + sum_reg(Var.D_Eant_SO2) + sum_reg(Var.D_Ebb_SO2)) - Par.Etot_SO2_pi)


## NH3-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_NH3', 
    In = ('D_Eant_NH3', 'D_Ebb_NH3'), 
    Eq = lambda Var, Par: Eq__ERF_ari_NH3(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_NH3(Var, Par):
    return Par.ph_ari_NH3 * (sum_reg(Var.D_Eant_NH3) + sum_reg(Var.D_Ebb_NH3))


## NOx-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_NOx', 
    In = ('D_Eant_NOx', 'D_Ebb_NOx'), 
    Eq = lambda Var, Par: Eq__ERF_ari_NOx(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_NOx(Var, Par):
    return Par.ph_ari_NOx * (sum_reg(Var.D_Eant_NOx) + sum_reg(Var.D_Ebb_NOx))


## VOC-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_VOC', 
    In = ('D_Eant_VOC', 'D_Ebb_VOC'), 
    Eq = lambda Var, Par: Eq__ERF_ari_VOC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_VOC(Var, Par):
    return Par.ph_ari_VOC * (sum_reg(Var.D_Eant_VOC) + sum_reg(Var.D_Ebb_VOC))


## CH4-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__ERF_ari_CH4(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_CH4(Var, Par):
    return Par.ph_ari_CH4 * Var.D_CH4


## N2O-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_N2O', 
    In = ('D_N2O',), 
    Eq = lambda Var, Par: Eq__ERF_ari_N2O(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_N2O(Var, Par):
    return Par.ph_ari_N2O * Var.D_N2O


## EESC-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_EESC', 
    In = ('D_EESC',), 
    Eq = lambda Var, Par: Eq__ERF_ari_EESC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_EESC(Var, Par):
    return Par.ph_ari_EESC * Var.D_EESC.sel(age_air='mid_lat', drop=True)


## H2-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_H2', 
    In = ('D_H2',), 
    Eq = lambda Var, Par: Eq__ERF_ari_H2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_H2(Var, Par):
    return Par.ph_ari_H2 * Var.D_H2


## mineral dust-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_dust', 
    In = ('D_Edust', 'D_Tg',), 
    Eq = lambda Var, Par: Eq__ERF_ari_dust(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_dust(Var, Par):
    return Par.ph_ari_dust * (safe_exp(Par.g_tau_dust * Var.D_Tg, 5.) * (Par.Edust_pi + Var.D_Edust) - Par.Edust_pi)


## sea salt-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_salt', 
    In = ('D_Esalt', 'D_Tg',), 
    Eq = lambda Var, Par: Eq__ERF_ari_salt(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_salt(Var, Par):
    return Par.ph_ari_salt * (safe_exp(Par.g_tau_salt * Var.D_Tg, 5.) * (Par.Esalt_pi + Var.D_Esalt) - Par.Esalt_pi)


## DMS-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_DMS', 
    In = ('D_Enat_DMS', 'D_Tg',), 
    Eq = lambda Var, Par: Eq__ERF_ari_DMS(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_DMS(Var, Par):
    return Par.ph_ari_DMS * (safe_exp(Par.g_tau_SO4 * Var.D_Tg, 5.) * (Par.Enat_DMS_pi + Var.D_Enat_DMS) - Par.Enat_DMS_pi)


## BVOC-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_BVOC', 
    In = ('D_Enat_BVOC',), 
    Eq = lambda Var, Par: Eq__ERF_ari_BVOC(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_BVOC(Var, Par):
    return Par.ph_ari_BVOC * Var.D_Enat_BVOC


## LNOx-induced aerosol-radiation interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari_LNOx', 
    In = ('D_Enat_LNOx',), 
    Eq = lambda Var, Par: Eq__ERF_ari_LNOx(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari_LNOx(Var, Par):
    return Par.ph_ari_LNOx * Var.D_Enat_LNOx


## aerosol-radiation interactions (aggregated)
OSCAR_ERF_slcf.process(
    Out = 'ERF_ari', 
    In = ('ERF_ari_BC', 'ERF_ari_OC', 'ERF_ari_SO2', 'ERF_ari_NH3', 'ERF_ari_NOx', 'ERF_ari_VOC', 'ERF_ari_CH4', 'ERF_ari_N2O', 'ERF_ari_EESC', 'ERF_ari_H2', 'ERF_ari_dust', 'ERF_ari_salt', 'ERF_ari_DMS', 'ERF_ari_BVOC', 'ERF_ari_LNOx'), 
    Eq = lambda Var, Par: Eq__ERF_ari(Var, Par), 
    units = 'W m-2')

def Eq__ERF_ari(Var, Par):
    return Var.ERF_ari_BC + Var.ERF_ari_OC + Var.ERF_ari_SO2 + Var.ERF_ari_NH3 + Var.ERF_ari_NOx + Var.ERF_ari_VOC + Var.ERF_ari_CH4 + Var.ERF_ari_N2O + Var.ERF_ari_EESC + Var.ERF_ari_H2 + Var.ERF_ari_dust + Var.ERF_ari_salt + Var.ERF_ari_DMS + Var.ERF_ari_BVOC + Var.ERF_ari_LNOx


## aerosol-cloud interactions
OSCAR_ERF_slcf.process(
    Out = 'ERF_aci', 
    In = ('D_Eant_SO2', 'D_Ebb_SO2', 'D_Eant_OC', 'D_Ebb_OC', 'D_Eant_BC', 'D_Ebb_BC'), 
    Eq = lambda Var, Par: Eq__ERF_aci(Var, Par), 
    units = 'W m-2')

def Eq__ERF_aci(Var, Par):
    D_Eaci_SO2 = sum_reg(Var.D_Eant_SO2) + sum_reg(Var.D_Ebb_SO2)
    D_Eaci_OC = sum_reg(Var.D_Eant_OC) + sum_reg(Var.D_Ebb_OC)
    D_Eaci_BC = sum_reg(Var.D_Eant_BC) + sum_reg(Var.D_Ebb_BC)
    return Par.Ph_aci * np.log1p(D_Eaci_SO2 / Par.Eaci_SO2 + D_Eaci_OC / Par.Eaci_OC  + D_Eaci_BC / Par.Eaci_BC)


## all aerosols effects (aggregated)
OSCAR_ERF_slcf.process(
    Out = 'ERF_aer', 
    In = ('ERF_ari', 'ERF_aci'), 
    Eq = lambda Var, Par: Eq__ERF_aer(Var, Par), 
    units = 'W m-2')

def Eq__ERF_aer(Var, Par):
    return Var.ERF_ari + Var.ERF_aci

