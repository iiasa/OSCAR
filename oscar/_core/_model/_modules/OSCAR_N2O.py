import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   NITROUS OXIDE BUDGET
#####################################################################

## initialize
OSCAR_N2O = Model('OSCAR_N2O')


##=====================
## Secondary parameters
##=====================

## preindustrial stratospheric sink rate
OSCAR_N2O.process(
    Out = 'v_N2O_hv', 
    Eq = lambda Par: Eq__v_N2O_hv(Par), 
    units = 'yr-1')

def Eq__v_N2O_hv(Par):
    if 'f_hv_pd' not in Par: return None
    return Par.v_N2O_hv_pd / Par.f_hv_pd


## preindustrial N2O lifetime
OSCAR_N2O.process(
    Out = 'tau_N2O_pi', 
    Eq = lambda Par: Eq__tau_N2O_pi(Par), 
    units = 'yr')

def Eq__tau_N2O_pi(Par):
    if 'v_N2O_hv' not in Par: return None
    return (Par.v_N2O_hv) ** -1


## preindustrial N2O sink (= unspecified emissions)
OSCAR_N2O.process(
    Out = 'Fsink_N2O_pi', 
    Eq = lambda Par: Eq__Fsink_N2O_pi(Par), 
    units = 'TgN yr-1')

def Eq__Fsink_N2O_pi(Par):
    if 'tau_N2O_pi' not in Par: return None
    else: return Par.a_N2O * Par.N2O_pi / Par.tau_N2O_pi


##=====================
## Diagnostic variables
##=====================

## N2O stratospheric sink
OSCAR_N2O.process(
    Out = 'D_Fhv_N2O', 
    In = ('f_hv', 'D_N2O'), 
    Eq = lambda Var, Par: Eq__D_Fhv_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Fhv_N2O(Var, Par):
    return Par.a_N2O * Par.v_N2O_hv * ((Par.N2O_pi + Var.D_N2O) * Var.f_hv - Par.N2O_pi)


## N2O total atmospheric sink
OSCAR_N2O.process(
    Out = 'D_Fsink_N2O', 
    In = ('D_Fhv_N2O',), 
    Eq = lambda Var, Par: Eq__D_Fsink_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Fsink_N2O(Var, Par):
    return Var.D_Fhv_N2O


## N2O missing emissions
## note: zero by default, can be used to prescribe any additional flux
OSCAR_N2O.process(
    Out = 'D_Emiss_N2O', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Emiss_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Emiss_N2O(Var, Par):
    return 0.


## atmospheric N2O budget
OSCAR_N2O.process(
    Out = 'd_N2O', 
    In = ('D_Eant_N2O', 'D_Ebb_N2O', 'D_Fsink_N2O', 'D_Emiss_N2O'), 
    Eq = lambda Var, Par: Eq__d_N2O(Var, Par),
    units = 'ppb yr-1')

def Eq__d_N2O(Var, Par):
    return 1 / Par.a_N2O * (sum_reg(Var.D_Eant_N2O) + sum_reg(Var.D_Ebb_N2O) - Var.D_Fsink_N2O + Var.D_Emiss_N2O)


## N2O residual emissions (= budget imbalance)
## note: non-zero only if d_N2O is prescribed!
OSCAR_N2O.process(
    Out = 'Eimb_N2O', 
    In = ('d_N2O', 'D_Eant_N2O', 'D_Ebb_N2O', 'D_Fsink_N2O', 'D_Emiss_N2O'), 
    Eq = lambda Var, Par: Eq__Eimb_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__Eimb_N2O(Var, Par):
    return Par.a_N2O * Var.d_N2O - (sum_reg(Var.D_Eant_N2O) + sum_reg(Var.D_Ebb_N2O) - Var.D_Fsink_N2O + Var.D_Emiss_N2O)


## N2O lifetime
OSCAR_N2O.process(
    Out = 'tau_N2O', 
    In = ('D_Fsink_N2O', 'D_N2O'), 
    Eq = lambda Var, Par: Eq__tau_N2O(Var, Par), 
    units = 'yr')

def Eq__tau_N2O(Var, Par):
    return Par.a_N2O * (Par.N2O_pi + Var.D_N2O) / (Par.Fsink_N2O_pi + Var.D_Fsink_N2O)


## total atmospheric N2O concentration
OSCAR_N2O.process(
    Out = 'N2O', 
    In = ('D_N2O',), 
    Eq = lambda Var, Par: Eq__N2O(Var, Par), 
    units = 'ppb')

def Eq__N2O(Var, Par):
    return Par.N2O_pi + Var.D_N2O


##=====================
## Prognostic variables
##=====================

## atmospheric N2O concentration
OSCAR_N2O.process(
    Out = 'D_N2O', 
    In = ('D_N2O', 'd_N2O'), 
    DiffEq = lambda Var, Par: DiffEq__D_N2O(Var, Par), 
    vLin = lambda Par: vLin__D_N2O(Par), 
    units = 'ppb')

def DiffEq__D_N2O(Var, Par):
    return Var.d_N2O

def vLin__D_N2O(Par):
    return 1 / Par.tau_N2O_pi

