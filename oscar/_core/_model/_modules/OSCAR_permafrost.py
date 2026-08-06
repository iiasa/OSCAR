import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, f_max


#####################################################################
##   PERMAFROST CARBON
#####################################################################

## initialize
OSCAR_permafrost = Model('OSCAR_permafrost')

## module taken from:
## (Gasser et al., 2018; https://doi.org/10.1038/s41561-018-0227-0)


##=====================
## Diagnostic variables
##=====================

## heterotrophic respiration factor for permafrost
OSCAR_permafrost.process(
    Out = 'r_ethaw', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__r_ethaw(Var, Par), 
    units = '1')

def Eq__r_ethaw(Var, Par):
    fct_T = safe_exp(Par.x_ethaw * Par.g_ethaw_T * Par.a_Tpf_Tg * Var.D_Tg,  f_max(1/Par.t_ethaw))
    fct_T2 = np.exp(-Par.x_ethaw * Par.g_ethaw_T2 * (Par.a_Tpf_Tg * Var.D_Tg)**2)
    return fct_T * fct_T2


## theoretical thawed fraction
OSCAR_permafrost.process(
    Out = 'D_pthaw_bar', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_pthaw_bar(Var, Par), 
    units = '1')

def Eq__D_pthaw_bar(Var, Par):
    return -Par.pthaw_min + (1 + Par.pthaw_min) / (1 + ((1/Par.pthaw_min + 1) ** Par.x_pthaw - 1) * np.exp(-Par.g_pthaw * Par.x_pthaw * Par.a_Tpf_Tg * Var.D_Tg)) ** (1/Par.x_pthaw)


## derivative of actual thawed fraction
OSCAR_permafrost.process(
    Out = 'd_pthaw', 
    In = ('D_pthaw_bar', 'D_pthaw'), 
    Eq = lambda Var, Par: Eq__d_pthaw(Var, Par), 
    units = 'yr-1')

def Eq__d_pthaw(Var, Par):
    # other way to formulate (because v_thaw > v_froz):
    # 0.5 * (Par.v_thaw + Par.v_froz) * (Var.D_pthaw_bar - Var.D_pthaw) + 0.5 * np.abs((Par.v_thaw - Par.v_froz) * (Var.D_pthaw_bar - Var.D_pthaw))
    return (Par.v_thaw * (Var.D_pthaw_bar >= Var.D_pthaw) + Par.v_froz * (Var.D_pthaw_bar < Var.D_pthaw)) * (Var.D_pthaw_bar - Var.D_pthaw)


## flux of thawing permafrost carbon
OSCAR_permafrost.process(
    Out = 'D_Fthaw', 
    In = ('d_pthaw',), 
    Eq = lambda Var, Par: Eq__D_Fthaw(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fthaw(Var, Par):
    return Par.Cfroz_pi * Var.d_pthaw


## emissions from thawed permafrost carbon
OSCAR_permafrost.process(
    Out = 'D_Ethaw', 
    In = ('D_Cthaw', 'r_ethaw'), 
    Eq = lambda Var, Par: Eq__D_Ethaw(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Ethaw(Var, Par):
    return 1 / Par.t_ethaw * Var.r_ethaw * Var.D_Cthaw


## total permafrost carbon emissions
OSCAR_permafrost.process(
    Out = 'D_Epf', 
    In = ('D_Ethaw', 'D_Fthaw'), 
    Eq = lambda Var, Par: Eq__D_Epf(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Epf(Var, Par):
    return Var.D_Ethaw.sum('box_thaw', min_count=1) + Par.p_pf_inst * Var.D_Fthaw


## CO2 permafrost emissions
OSCAR_permafrost.process(
    Out = 'D_Epf_CO2', 
    In = ('D_Epf',), 
    Eq = lambda Var, Par: Eq__D_Epf_CO2(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Epf_CO2(Var, Par):
    return (1 - Par.p_pf_CH4) * Var.D_Epf.sum('reg_pf', min_count=1)


## CH4 permafrost emissions
OSCAR_permafrost.process(
    Out = 'D_Epf_CH4', 
    In = ('D_Epf',), 
    Eq = lambda Var, Par: Eq__D_Epf_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Epf_CH4(Var, Par):
    return Par.Pg_to_Tg * Par.p_pf_CH4 * Var.D_Epf.sum('reg_pf', min_count=1)


## ADDITIONAL DIAGNOSTICS

## total permafrost carbon pool
OSCAR_permafrost.process(
    Out = 'Cperm', 
    In = ('D_Cfroz', 'D_Cthaw'), 
    Eq = lambda Var, Par: Eq__Cperm(Var, Par), 
    units = 'PgC')

def Eq__Cperm(Var, Par):
    return Par.Cfroz_pi + Var.D_Cfroz +  Var.D_Cthaw.sum('box_thaw', min_count=1)


##=====================
## Prognostic variables
##=====================

## actual thawed fraction
OSCAR_permafrost.process(
    Out = 'D_pthaw', 
    In = ('D_pthaw', 'd_pthaw'), 
    DiffEq = lambda Var, Par: DiffEq__D_pthaw(Var, Par), 
    vLin = lambda Par: vLin__D_pthaw(Par), 
    units = '1', 
    core_dims = ['reg_pf'])

def DiffEq__D_pthaw(Var, Par):
    return Var.d_pthaw

def vLin__D_pthaw(Par):
    return 0.5 * (Par.v_thaw + Par.v_froz)


## frozen permafrost carbon
OSCAR_permafrost.process(
    Out = 'D_Cfroz', 
    In = ('D_Cfroz', 'D_Fthaw'), 
    DiffEq = lambda Var, Par: DiffEq__D_Cfroz(Var, Par), 
    units = 'PgC', 
    core_dims = ['reg_pf'])

def DiffEq__D_Cfroz(Var, Par):
    return -Var.D_Fthaw


## thawed permafrost carbon
OSCAR_permafrost.process(
    Out = 'D_Cthaw', 
    In = ('D_Cthaw', 'D_Fthaw', 'D_Ethaw'), 
    DiffEq = lambda Var, Par: DiffEq__D_Cthaw(Var, Par), 
    vLin = lambda Par: vLin__D_Cthaw(Par), 
    units = 'PgC', 
    core_dims = ['reg_pf', 'box_thaw'])

def DiffEq__D_Cthaw(Var, Par):
    return Par.p_fthaw * (1 - Par.p_pf_inst) * Var.D_Fthaw - Var.D_Ethaw

def vLin__D_Cthaw(Par):
    return 1 / Par.t_ethaw

