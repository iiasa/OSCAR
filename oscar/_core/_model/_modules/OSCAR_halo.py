import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   HALOGENATED COMPOUNDS BUDGET
#####################################################################

## initialize
OSCAR_halo = Model('OSCAR_halo')


##=====================
## Secondary parameters
##=====================

## preindustrial tropospheric hydroxyl sink rate
OSCAR_halo.process(
    Out = 'v_Xhalo_OH', 
    Eq = lambda Par: Eq__v_Xhalo_OH(Par), 
    units = 'yr-1')

def Eq__v_Xhalo_OH(Par):
    if 'f_kOH_pd' not in Par: return None
    return Par.v_Xhalo_OH_pd / Par.f_kOH_pd


## preindustrial stratospheric sink rate
OSCAR_halo.process(
    Out = 'v_Xhalo_hv', 
    Eq = lambda Par: Eq__v_Xhalo_hv(Par), 
    units = 'yr-1')

def Eq__v_Xhalo_hv(Par):
    if 'f_hv_pd' not in Par: return None
    return Par.v_Xhalo_hv_pd / Par.f_hv_pd


## halogenated compounds preindustrial lifetime
OSCAR_halo.process(
    Out = 'tau_Xhalo_pi', 
    Eq = lambda Par: Eq__tau_Xhalo_pi(Par), 
    units = 'yr', 
    core_dims=['spc_halo'])

def Eq__tau_Xhalo_pi(Par):
    if 'v_Xhalo_OH' not in Par: return None
    if 'v_Xhalo_hv' not in Par: return None
    return (Par.v_Xhalo_OH + Par.v_Xhalo_hv + Par.v_Xhalo_other) ** -1


## halogenated compounds preindustrial sink (= unspecified emissions)
OSCAR_halo.process(
    Out = 'Fsink_Xhalo_pi', 
    Eq = lambda Par: Eq__Fsink_Xhalo_pi(Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__Fsink_Xhalo_pi(Par):
    if 'tau_Xhalo_pi' not in Par: return None
    else: return Par.a_Xhalo * Par.Xhalo_pi / Par.tau_Xhalo_pi


##=====================
## Diagnostic variables
##=====================

## halogenated compounds tropospheric hydroxyl sink
OSCAR_halo.process(
    Out = 'D_Foh_Xhalo', 
    In = ('f_kOH', 'D_Xhalo'), 
    Eq = lambda Var, Par: Eq__D_Foh_Xhalo(Var, Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__D_Foh_Xhalo(Var, Par):
    return Par.a_Xhalo * Par.v_Xhalo_OH * ((Par.Xhalo_pi + Var.D_Xhalo) * Var.f_kOH - Par.Xhalo_pi)


## halogenated compounds stratospheric sink
OSCAR_halo.process(
    Out = 'D_Fhv_Xhalo', 
    In = ('f_hv', 'D_Xhalo'), 
    Eq = lambda Var, Par: Eq__D_Fhv_Xhalo(Var, Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__D_Fhv_Xhalo(Var, Par):
    return Par.a_Xhalo * Par.v_Xhalo_hv * ((Par.Xhalo_pi + Var.D_Xhalo) * Var.f_hv - Par.Xhalo_pi)


## halogenated compounds tropospheric other sinks combined
OSCAR_halo.process(
    Out = 'D_Fother_Xhalo', 
    In = ('D_Xhalo',), 
    Eq = lambda Var, Par: Eq__D_Fother_Xhalo(Var, Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__D_Fother_Xhalo(Var, Par):
    return Par.a_Xhalo * Par.v_Xhalo_other * Var.D_Xhalo


## halogenated compounds total atmospheric sink
OSCAR_halo.process(
    Out = 'D_Fsink_Xhalo', 
    In = ('D_Foh_Xhalo', 'D_Fhv_Xhalo', 'D_Fother_Xhalo'), 
    Eq = lambda Var, Par: Eq__D_Fsink_Xhalo(Var, Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__D_Fsink_Xhalo(Var, Par):
    return Var.D_Foh_Xhalo + Var.D_Fhv_Xhalo + Var.D_Fother_Xhalo


## halogenated compounds missing emissions
## note: zero by default, can be used to prescribe any additional flux
OSCAR_halo.process(
    Out = 'D_Emiss_Xhalo', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Emiss_Xhalo(Var, Par), 
    units = 'Gg yr-1')

def Eq__D_Emiss_Xhalo(Var, Par):
    return 0.


## halogenated compounds atmospheric budget
OSCAR_halo.process(
    Out = 'd_Xhalo', 
    In = ('D_Eant_Xhalo', 'D_Fsink_Xhalo', 'D_Emiss_Xhalo'), 
    Eq = lambda Var, Par: Eq__d_Xhalo(Var, Par),
    units = 'ppt yr-1', 
    core_dims=['spc_halo'])

def Eq__d_Xhalo(Var, Par):
    return 1 / Par.a_Xhalo * (sum_reg(Var.D_Eant_Xhalo) - Var.D_Fsink_Xhalo + Var.D_Emiss_Xhalo)


## halogenated compounds residual emissions (= budget imbalance)
## note: non-zero only if d_Xhalo is prescribed!
OSCAR_halo.process(
    Out = 'Eimb_Xhalo', 
    In = ('d_Xhalo', 'D_Eant_Xhalo', 'D_Fsink_Xhalo', 'D_Emiss_Xhalo'), 
    Eq = lambda Var, Par: Eq__Eimb_Xhalo(Var, Par), 
    units = 'Gg yr-1', 
    core_dims=['spc_halo'])

def Eq__Eimb_Xhalo(Var, Par):
    return Par.a_Xhalo * Var.d_Xhalo - (sum_reg(Var.D_Eant_Xhalo) - Var.D_Fsink_Xhalo + Var.D_Emiss_Xhalo)


## halogenated compounds lifetime
OSCAR_halo.process(
    Out = 'tau_Xhalo', 
    In = ('D_Fsink_Xhalo', 'D_Xhalo'), 
    Eq = lambda Var, Par: Eq__tau_Xhalo(Var, Par), 
    units = 'yr', 
    core_dims=['spc_halo'])

def Eq__tau_Xhalo(Var, Par):
    return Par.a_Xhalo * (Par.Xhalo_pi + Var.D_Xhalo) / (Par.Fsink_Xhalo_pi + Var.D_Fsink_Xhalo)


## total halogenated compounds atmospheric concentration
OSCAR_halo.process(
    Out = 'Xhalo', 
    In = ('D_Xhalo',), 
    Eq = lambda Var, Par: Eq__Xhalo(Var, Par), 
    units = 'ppt')

def Eq__Xhalo(Var, Par):
    return Par.Xhalo_pi + Var.D_Xhalo


##=====================
## Prognostic variables
##=====================

## halogenated compounds atmospheric concentration
OSCAR_halo.process(
    Out = 'D_Xhalo', 
    In = ('D_Xhalo', 'd_Xhalo'), 
    DiffEq = lambda Var, Par: DiffEq__D_Xhalo(Var, Par), 
    vLin = lambda Par: vLin__D_Xhalo(Par), 
    units = 'ppt', 
    core_dims=['spc_halo'])

def DiffEq__D_Xhalo(Var, Par):
    return Var.d_Xhalo

def vLin__D_Xhalo(Par):
    return 1 / Par.tau_Xhalo_pi


## halogenated compounds lagged concentration
## note: this is a linearized version of the delayed equation
OSCAR_halo.process(
    Out = 'D_Xhalo_lag', 
    In = ('D_Xhalo_lag', 'D_Xhalo'), 
    DiffEq = lambda Var, Par: DiffEq__D_Xhalo_lag(Var, Par), 
    vLin = lambda Par: vLin__D_Xhalo_lag(Par), 
    units = 'ppt', 
    core_dims=['spc_halo', 'age_air'])

def DiffEq__D_Xhalo_lag(Var, Par):
    return 1 / Par.t_lag * (Var.D_Xhalo - Var.D_Xhalo_lag)

def vLin__D_Xhalo_lag(Par):
    return 1 / Par.t_lag

