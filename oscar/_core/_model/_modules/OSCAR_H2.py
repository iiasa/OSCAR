import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   HYDROGEN BUDGET
#####################################################################

## initialize
OSCAR_H2 = Model('OSCAR_H2')

## module published in:
## (Ouyang et al., 2025; https://doi.org/10.1038/s41586-025-09806-1)


##=====================
## Secondary parameters
##=====================

## preindustrial tropospheric hydroxyl sink rate
OSCAR_H2.process(
    Out = 'v_H2_OH', 
    Eq = lambda Par: Eq__v_H2_OH(Par), 
    units = 'yr-1')

def Eq__v_H2_OH(Par):
    if 'r_kOH_pd' not in Par: return None
    return Par.v_H2_OH_pd / Par.r_kOH_pd


## preindustrial H2 lifetime
OSCAR_H2.process(
    Out = 'tau_H2_pi', 
    Eq = lambda Par: Eq__tau_H2_pi(Par), 
    units = 'yr')

def Eq__tau_H2_pi(Par):
    if 'v_H2_OH' not in Par: return None
    return (Par.v_H2_OH + Par.v_H2_soil) ** -1


## preindustrial H2 sink (= unspecified emissions)
OSCAR_H2.process(
    Out = 'Fsink_H2_pi', 
    Eq = lambda Par: Eq__Fsink_H2_pi(Par), 
    units = 'TgH2 yr-1')

def Eq__Fsink_H2_pi(Par):
    if 'tau_H2_pi' not in Par: return None
    else: return Par.a_H2 * Par.H2_pi / Par.tau_H2_pi


##=====================
## Diagnostic variables
##=====================

## H2 photochemical production through CH4 oxidation
OSCAR_H2.process(
    Out = 'D_Fprod_H2_CH4', 
    In = ('D_Foh_CH4',), 
    Eq = lambda Var, Par: Eq__D_Fprod_H2_CH4(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Fprod_H2_CH4(Var, Par):
    return Par.ch_H2_CH4 * Var.D_Foh_CH4


## H2 photochemical production through VOCs oxidation
OSCAR_H2.process(
    Out = 'D_Fprod_H2_VOC', 
    In = ('D_Eant_VOC', 'D_Ebb_VOC', 'D_Enat_BVOC'), 
    Eq = lambda Var, Par: Eq__D_Fprod_H2_VOC(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Fprod_H2_VOC(Var, Par):
    return Par.ch_H2_VOC_ant * sum_reg(Var.D_Eant_VOC) + Par.ch_H2_VOC_bb * sum_reg(Var.D_Ebb_VOC) + Par.ch_H2_BVOC * Var.D_Enat_BVOC

## H2 total photochemical production
OSCAR_H2.process(
    Out = 'D_Fprod_H2', 
    In = ('D_Fprod_H2_CH4',  'D_Fprod_H2_VOC'), 
    Eq = lambda Var, Par: Eq__D_Fprod_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Fprod_H2(Var, Par):
    return Var.D_Fprod_H2_CH4 + Var.D_Fprod_H2_VOC


## H2 hydroxyl sink
OSCAR_H2.process(
    Out = 'D_Foh_H2', 
    In = ('r_kOH', 'D_H2'), 
    Eq = lambda Var, Par: Eq__D_Foh_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Foh_H2(Var, Par):
    return Par.a_H2 * Par.v_H2_OH * ((Par.H2_pi + Var.D_H2) * Var.r_kOH - Par.H2_pi)


## H2 soil sink
OSCAR_H2.process(
    Out = 'D_Fsoil_H2', 
    In = ('D_H2',), 
    Eq = lambda Var, Par: Eq__D_Fsoil_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Fsoil_H2(Var, Par):
    return Par.a_H2 * Par.v_H2_soil * Var.D_H2


## H2 total atmospheric sink
OSCAR_H2.process(
    Out = 'D_Fsink_H2', 
    In = ('D_Foh_H2', 'D_Fsoil_H2'), 
    Eq = lambda Var, Par: Eq__D_Fsink_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Fsink_H2(Var, Par):
    return Var.D_Foh_H2 + Var.D_Fsoil_H2


## H2 missing emissions
## note: zero by default, can be used to prescribe any additional flux
OSCAR_H2.process(
    Out = 'D_Emiss_H2', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Emiss_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Emiss_H2(Var, Par):
    return 0.


## atmospheric H2 budget
OSCAR_H2.process(
    Out = 'd_H2', 
    In = ('D_Eant_H2', 'D_Ebb_H2', 'D_Fprod_H2', 'D_Fsink_H2', 'D_Emiss_H2'), 
    Eq = lambda Var, Par: Eq__d_H2(Var, Par),
    units = 'ppb yr-1')

def Eq__d_H2(Var, Par):
    return 1 / Par.a_H2 * (sum_reg(Var.D_Eant_H2) + sum_reg(Var.D_Ebb_H2) + Var.D_Fprod_H2 - Var.D_Fsink_H2 + Var.D_Emiss_H2)


## ADDITIONAL DIAGNOSTICS

## H2 residual emissions (= budget imbalance)
## note: non-zero only if d_H2 is prescribed!
OSCAR_H2.process(
    Out = 'Eimb_H2', 
    In = ('d_H2', 'D_Eant_H2', 'D_Ebb_H2', 'D_Fprod_H2', 'D_Fsink_H2', 'D_Emiss_H2'), 
    Eq = lambda Var, Par: Eq__Eimb_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__Eimb_H2(Var, Par):
    return Par.a_H2 * Var.d_H2 - (sum_reg(Var.D_Eant_H2) + sum_reg(Var.D_Ebb_H2) + Var.D_Fprod_H2 - Var.D_Fsink_H2 + Var.D_Emiss_H2)


## H2 lifetime
OSCAR_H2.process(
    Out = 'tau_H2', 
    In = ('D_Fsink_H2', 'D_H2'), 
    Eq = lambda Var, Par: Eq__tau_H2(Var, Par), 
    units = 'yr')

def Eq__tau_H2(Var, Par):
    return Par.a_H2 * (Par.H2_pi + Var.D_H2) / (Par.Fsink_H2_pi + Var.D_Fsink_H2)


## total atmospheric H2 concentration
OSCAR_H2.process(
    Out = 'H2', 
    In = ('D_H2',), 
    Eq = lambda Var, Par: Eq__H2(Var, Par), 
    units = 'ppb')

def Eq__H2(Var, Par):
    return Par.H2_pi + Var.D_H2


##=====================
## Prognostic variables
##=====================

## atmospheric H2 concentration
OSCAR_H2.process(
    Out = 'D_H2', 
    In = ('D_H2', 'd_H2'), 
    DiffEq = lambda Var, Par: DiffEq__D_H2(Var, Par), 
    vLin = lambda Par: vLin__D_H2(Par), 
    units = 'ppb')

def DiffEq__D_H2(Var, Par):
    return Var.d_H2

def vLin__D_H2(Par):
    return 1 / Par.tau_H2_pi

