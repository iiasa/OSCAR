import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   METHANE BUDGET
#####################################################################

## initialize
OSCAR_CH4 = Model('OSCAR_CH4')


##=====================
## Secondary parameters
##=====================

## preindustrial tropospheric hydroxyl sink rate
OSCAR_CH4.process(
    Out = 'v_CH4_OH', 
    Eq = lambda Par: Eq__v_CH4_OH(Par), 
    units = 'yr-1')

def Eq__v_CH4_OH(Par):
    if 'f_kOH_pd' not in Par: return None
    return Par.v_CH4_OH_pd / Par.f_kOH_pd


## preindustrial stratospheric sink rate
OSCAR_CH4.process(
    Out = 'v_CH4_hv', 
    Eq = lambda Par: Eq__v_CH4_hv(Par), 
    units = 'yr-1')

def Eq__v_CH4_hv(Par):
    if 'f_hv_pd' not in Par: return None
    return Par.v_CH4_hv_pd / Par.f_hv_pd


## preindustrial CH4 lifetime
OSCAR_CH4.process(
    Out = 'tau_CH4_pi', 
    Eq = lambda Par: Eq__tau_CH4_pi(Par), 
    units = 'yr')

def Eq__tau_CH4_pi(Par):
    if 'v_CH4_OH' not in Par: return None
    if 'v_CH4_hv' not in Par: return None
    return (Par.v_CH4_OH + Par.v_CH4_hv + Par.v_CH4_soil + Par.v_CH4_Cl) ** -1


## preindustrial CH4 sink (= unspecified emissions)
OSCAR_CH4.process(
    Out = 'Fsink_CH4_pi', 
    Eq = lambda Par: Eq__Fsink_CH4_pi(Par), 
    units = 'TgC yr-1')

def Eq__Fsink_CH4_pi(Par):
    if 'tau_CH4_pi' not in Par: return None
    return Par.a_CH4 * Par.CH4_pi / Par.tau_CH4_pi


##=====================
## Diagnostic variables
##=====================

## CH4 tropospheric hydroxyl sink
OSCAR_CH4.process(
    Out = 'D_Foh_CH4', 
    In = ('f_kOH', 'D_CH4'), 
    Eq = lambda Var, Par: Eq__D_Foh_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Foh_CH4(Var, Par):
    return Par.a_CH4 * Par.v_CH4_OH * ((Par.CH4_pi + Var.D_CH4) * Var.f_kOH - Par.CH4_pi)


## CH4 stratospheric sink
OSCAR_CH4.process(
    Out = 'D_Fhv_CH4', 
    In = ('f_hv', 'D_CH4'), 
    Eq = lambda Var, Par: Eq__D_Fhv_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Fhv_CH4(Var, Par):
    return Par.a_CH4 * Par.v_CH4_hv * ((Par.CH4_pi + Var.D_CH4) * Var.f_hv - Par.CH4_pi)


## CH4 tropospheric chlorine sink (in oceanic boundary layer)
OSCAR_CH4.process(
    Out = 'D_Fcl_CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__D_Fcl_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Fcl_CH4(Var, Par):
    return Par.a_CH4 * Par.v_CH4_Cl * Var.D_CH4


## CH4 dry soil sink (= methanotrophy)
OSCAR_CH4.process(
    Out = 'D_Fsoil_CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__D_Fsoil_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Fsoil_CH4(Var, Par):
    return Par.a_CH4 * Par.v_CH4_soil * Var.D_CH4


## CH4 total atmospheric sink
OSCAR_CH4.process(
    Out = 'D_Fsink_CH4', 
    In = ('D_Foh_CH4', 'D_Fhv_CH4', 'D_Fcl_CH4', 'D_Fsoil_CH4'), 
    Eq = lambda Var, Par: Eq__D_Fsink_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Fsink_CH4(Var, Par):
    return Var.D_Foh_CH4 + Var.D_Fhv_CH4 + Var.D_Fcl_CH4 + Var.D_Fsoil_CH4


## CH4 missing emissions
## note: zero by default, can be used to prescribe any additional flux
OSCAR_CH4.process(
    Out = 'D_Emiss_CH4', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Emiss_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Emiss_CH4(Var, Par):
    return 0.


## atmospheric CH4 budget
OSCAR_CH4.process(
    Out = 'd_CH4', 
    In = ('D_Eant_CH4', 'D_Ebb_CH4', 'D_Ewet_CH4', 'D_Epf_CH4', 'D_Fsink_CH4', 'D_Emiss_CH4'), 
    Eq = lambda Var, Par: Eq__d_CH4(Var, Par),
    units = 'ppb yr-1')

def Eq__d_CH4(Var, Par):
    return 1 / Par.a_CH4 * (sum_reg(Var.D_Eant_CH4) + sum_reg(Var.D_Ebb_CH4) + Var.D_Ewet_CH4 + Var.D_Epf_CH4 - Var.D_Fsink_CH4 + Var.D_Emiss_CH4)


## CH4 residual emissions (= budget imbalance)
## note: non-zero only if d_CH4 is prescribed!
OSCAR_CH4.process(
    Out = 'Eimb_CH4', 
    In = ('d_CH4', 'D_Eant_CH4', 'D_Ebb_CH4', 'D_Ewet_CH4', 'D_Epf_CH4', 'D_Fsink_CH4', 'D_Emiss_CH4'), 
    Eq = lambda Var, Par: Eq__Eimb_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__Eimb_CH4(Var, Par):
    return Par.a_CH4 * Var.d_CH4 - (sum_reg(Var.D_Eant_CH4) + sum_reg(Var.D_Ebb_CH4) + Var.D_Ewet_CH4 + Var.D_Epf_CH4 - Var.D_Fsink_CH4 + Var.D_Emiss_CH4)


## CH4 lifetime
OSCAR_CH4.process(
    Out = 'tau_CH4', 
    In = ('D_Fsink_CH4', 'D_CH4'), 
    Eq = lambda Var, Par: Eq__tau_CH4(Var, Par), 
    units = 'yr')

def Eq__tau_CH4(Var, Par):
    return Par.a_CH4 * (Par.CH4_pi + Var.D_CH4) / (Par.Fsink_CH4_pi + Var.D_Fsink_CH4)


## total atmospheric CH4 concentration
OSCAR_CH4.process(
    Out = 'CH4', 
    In = ('D_CH4',), 
    Eq = lambda Var, Par: Eq__CH4(Var, Par), 
    units = 'ppb')

def Eq__CH4(Var, Par):
    return Par.CH4_pi + Var.D_CH4


##=====================
## Prognostic variables
##=====================

## atmospheric CH4 concentration
OSCAR_CH4.process(
    Out = 'D_CH4', 
    In = ('D_CH4', 'd_CH4'), 
    DiffEq = lambda Var, Par: DiffEq__D_CH4(Var, Par), 
    vLin = lambda Par: vLin__D_CH4(Par), 
    units = 'ppb')

def DiffEq__D_CH4(Var, Par):
    return Var.d_CH4

def vLin__D_CH4(Par):
    return 1 / Par.tau_CH4_pi

