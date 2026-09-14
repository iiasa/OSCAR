import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp, f_max, safe_ratio


##################################################
##   OSCAR (RAD variant)
##################################################

## import and copy main model
from oscar._core._model.OSCAR import OSCAR
OSCAR_rad = OSCAR.copy(new_name='OSCAR_rad')


## RAD PROCESSES

## ingoing flux to the ocean
OSCAR_rad.process(
    Out = 'D_Fin', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Fin(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fin(Var, Par):
    return Par.v_fg * Par.a_CO2 * 0.


## net primary productivity factor
OSCAR_rad.process(
    Out = 'r_npp', 
    In = ('D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_npp(Var, Par), 
    units = '1')

def Eq__r_npp(Var, Par):
    fct_Tl = safe_exp(Par.g_npp_T * Var.D_Tl, 100) * np.exp(-Par.g_npp_T2 * np.maximum(Var.D_Tl, 0)**2)
    fct_Pl = safe_exp(Par.x_npp_P * np.log(Var.r_Pl), 100)
    return safe_ratio(fct_Tl * fct_Pl)


## relative change in wildfire rate
OSCAR_rad.process(
    Out = 'r_vfire', 
    In = ('r_npp', 'D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__r_vfire(Var, Par), 
    units = '1')

def Eq__r_vfire(Var, Par):
    fct_npp = safe_exp(Par.x_fire_npp * np.log(Var.r_npp), f_max(Par.v_fire))
    fct_Tl = safe_exp(Par.g_fire_T * Var.D_Tl, f_max(Par.v_fire))
    fct_Pl = safe_exp(Par.g_fire_P * Par.Pl_pi * (Var.r_Pl - 1), f_max(Par.v_fire))
    return fct_npp * fct_Tl * fct_Pl


## wetland areal emissions
OSCAR_rad.process(
    Out = 'D_ewet', 
    In = ('D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__D_ewet(Var, Par), 
    units = 'TgC Mha-1 yr-1')

def Eq__D_ewet(Var, Par):
    fct_Tl = safe_exp(Par.g_ewet_T * Var.D_Tl, 5.)
    fct_Pl = safe_exp(Par.x_ewet_P * np.log(Var.r_Pl), 5.)
    return Par.ewet_pi * (safe_ratio(fct_Tl * fct_Pl)  - 1)
    

## wetland extent
OSCAR_rad.process(
    Out = 'D_Awet', 
    In = ('D_Tl', 'r_Pl'), 
    Eq = lambda Var, Par: Eq__D_Awet(Var, Par), 
    units = 'Mha')

def Eq__D_Awet(Var, Par):
    fct_Tl = safe_exp(Par.x_Awet_T * np.log1p(Var.D_Tl / Par.Tl_pi), 5.)
    fct_Pl = safe_exp(Par.x_Awet_P * np.log(Var.r_Pl), 5.)
    return Par.Awet_pi * (safe_ratio(fct_Tl * fct_Pl) - 1)

