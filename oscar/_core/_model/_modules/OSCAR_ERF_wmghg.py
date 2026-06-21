import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   EFFECTIVE RADIATIVE FORCING - WELL-MIXED GREENHOUSE GASES
#####################################################################

## initialize
OSCAR_ERF_wmghg = Model('OSCAR_ERF_wmghg')

## module adapted from:
## (Etminan et al., 2016; https://doi.org/10.1002/2016GL071930)


##=====================
## Secondary parameters
##=====================

## preindustrial CO2 stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_CO2_pi', 
    Eq = lambda Par: Eq__SARF_CO2_pi(Par), 
    units = 'W m-2')

def Eq__SARF_CO2_pi(Par):
    SARF = Par.Ph_CO2 / Par.x_rf_CO2 * ((Par.CO2_pi / Par.CO2_rf) ** Par.x_rf_CO2 - 1)
    overlap = Par.i_CO2_N2O / Par.x_rf_CO2_N2O * ((Par.N2O_pi / Par.N2O_rf) ** Par.x_rf_CO2_N2O - 1)
    return SARF * (1 + overlap)


## preindustrial CH4 stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_CH4_pi', 
    Eq = lambda Par: Eq__SARF_CH4_pi(Par), 
    units = 'W m-2')

def Eq__SARF_CH4_pi(Par):
    SARF = Par.Ph_CH4 / Par.x_rf_CH4 * ((Par.CH4_pi / Par.CH4_rf) ** Par.x_rf_CH4 - 1)
    overlap = Par.i_CH4_N2O / Par.x_rf_CH4_N2O * ((Par.N2O_pi / Par.N2O_rf) ** Par.x_rf_CH4_N2O - 1)
    return SARF * (1 + overlap)


## preindustrial N2O stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_N2O_pi', 
    Eq = lambda Par: Eq__SARF_N2O_pi(Par), 
    units = 'W m-2')

def Eq__SARF_N2O_pi(Par):
    SARF = Par.Ph_N2O / Par.x_rf_N2O * ((Par.N2O_pi / Par.N2O_rf) ** Par.x_rf_N2O - 1)
    overlap1 = Par.i_N2O_CO2 / Par.x_rf_N2O_CO2 * ((Par.CO2_pi / Par.CO2_rf) ** Par.x_rf_N2O_CO2 - 1)
    overlap2 = Par.i_N2O_CH4 / Par.x_rf_N2O_CH4 * ((Par.CH4_pi / Par.CH4_rf) ** Par.x_rf_N2O_CH4 - 1)
    return SARF * (1 + overlap1 + overlap2)


## radiative forcing at CO2 doubling
OSCAR_ERF_wmghg.process(
    Out = 'ERF_CO2_2x', 
    Eq = lambda Par: Eq__ERF_CO2_2x(Par), 
    units = 'W m-2')

def Eq__ERF_CO2_2x(Par):
    if 'SARF_CO2_pi' not in Par: return None
    SARF = Par.Ph_CO2 / Par.x_rf_CO2 * ((2 * Par.CO2_pi / Par.CO2_rf) ** Par.x_rf_CO2 - 1)
    overlap = Par.i_CO2_N2O / Par.x_rf_CO2_N2O * ((Par.N2O_pi / Par.N2O_rf) ** Par.x_rf_CO2_N2O - 1)
    return Par.a_adj_CO2 * (SARF * (1 + overlap) - Par.SARF_CO2_pi)


##=====================
## Diagnostic variables
##=====================

## CO2 stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_CO2', 
    In = ('D_CO2', 'D_N2O'), 
    Eq = lambda Var, Par: Eq__SARF_CO2(Var, Par), 
    units = 'W m-2')

def Eq__SARF_CO2(Var, Par):
    SARF = Par.Ph_CO2 / Par.x_rf_CO2 * (((Par.CO2_pi + Var.D_CO2) / Par.CO2_rf) ** Par.x_rf_CO2 - 1)
    overlap = Par.i_CO2_N2O / Par.x_rf_CO2_N2O * (((Par.N2O_pi + Var.D_N2O) / Par.N2O_rf) ** Par.x_rf_CO2_N2O - 1)
    return SARF * (1 + overlap) - Par.SARF_CO2_pi


## CH4 stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_CH4', 
    In = ('D_CH4', 'D_N2O'), 
    Eq = lambda Var, Par: Eq__SARF_CH4(Var, Par), 
    units = 'W m-2')

def Eq__SARF_CH4(Var, Par):
    SARF = Par.Ph_CH4 / Par.x_rf_CH4 * (((Par.CH4_pi + Var.D_CH4) / Par.CH4_rf) ** Par.x_rf_CH4 - 1)
    overlap = Par.i_CH4_N2O / Par.x_rf_CH4_N2O * (((Par.N2O_pi + Var.D_N2O) / Par.N2O_rf) ** Par.x_rf_CH4_N2O - 1)
    return SARF * (1 + overlap) - Par.SARF_CH4_pi


## N2O stratospheric-adjusted radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'SARF_N2O', 
    In = ('D_CO2', 'D_CH4', 'D_N2O'), 
    Eq = lambda Var, Par: Eq__SARF_N2O(Var, Par), 
    units = 'W m-2')

def Eq__SARF_N2O(Var, Par):
    SARF = Par.Ph_N2O / Par.x_rf_N2O * (((Par.N2O_pi + Var.D_N2O) / Par.N2O_rf) ** Par.x_rf_N2O - 1)
    overlap1 = Par.i_N2O_CO2 / Par.x_rf_N2O_CO2 * (((Par.CO2_pi + Var.D_CO2) / Par.CO2_rf) ** Par.x_rf_N2O_CO2 - 1)
    overlap2 = Par.i_N2O_CH4 / Par.x_rf_N2O_CH4 * (((Par.CH4_pi + Var.D_CH4) / Par.CH4_rf) ** Par.x_rf_N2O_CH4 - 1)
    return SARF * (1 + overlap1 + overlap2) - Par.SARF_N2O_pi


## CO2 effective radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'ERF_CO2', 
    In = ('SARF_CO2',), 
    Eq = lambda Var, Par: Eq__ERF_CO2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_CO2(Var, Par):
    return Par.a_adj_CO2 * Var.SARF_CO2


## CH4 effective radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'ERF_CH4', 
    In = ('SARF_CH4',), 
    Eq = lambda Var, Par: Eq__ERF_CH4(Var, Par), 
    units = 'W m-2')

def Eq__ERF_CH4(Var, Par):
    return Par.a_adj_CH4 * Var.SARF_CH4


## N2O effective radiative forcing
OSCAR_ERF_wmghg.process(
    Out = 'ERF_N2O', 
    In = ('SARF_N2O',), 
    Eq = lambda Var, Par: Eq__ERF_N2O(Var, Par), 
    units = 'W m-2')

def Eq__ERF_N2O(Var, Par):
    return Par.a_adj_N2O * Var.SARF_N2O


## halogenated compounds radiative forcing (split)
OSCAR_ERF_wmghg.process(
    Out = 'ERF_Xhalo', 
    In = ('D_Xhalo',), 
    Eq = lambda Var, Par: Eq__ERF_Xhalo(Var, Par), 
    units = 'W m-2')

def Eq__ERF_Xhalo(Var, Par):
    return Par.k_ph_halo * Par.ph_Xhalo * Var.D_Xhalo


## halogenated compounds radiative forcing (aggregated)
OSCAR_ERF_wmghg.process(
    Out = 'ERF_halo', 
    In = ('ERF_Xhalo',), 
    Eq = lambda Var, Par: Eq__ERF_halo(Var, Par), 
    units = 'W m-2')

def Eq__ERF_halo(Var, Par):
    return Var.ERF_Xhalo.sum('spc_halo', min_count=1)


## all non-CO2 greenhouse gases (aggregated)
OSCAR_ERF_wmghg.process(
    Out = 'ERF_nonCO2', 
    In = ('ERF_CH4', 'ERF_N2O', 'ERF_halo'), 
    Eq = lambda Var, Par: Eq__ERF_nonCO2(Var, Par), 
    units = 'W m-2')

def Eq__ERF_nonCO2(Var, Par):
    return Var.ERF_CH4 + Var.ERF_N2O + Var.ERF_halo


## all well-mixed greenhouse gases (aggregated)
OSCAR_ERF_wmghg.process(
    Out = 'ERF_wmghg', 
    In = ('ERF_CO2', 'ERF_nonCO2'), 
    Eq = lambda Var, Par: Eq__ERF_wmghg(Var, Par), 
    units = 'W m-2')

def Eq__ERF_wmghg(Var, Par):
    return Var.ERF_CO2 + Var.ERF_nonCO2

