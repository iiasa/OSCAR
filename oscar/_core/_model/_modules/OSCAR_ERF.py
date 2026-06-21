import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   EFFECTIVE RADIATIVE FORCING
#####################################################################

## import submodules
from oscar._core._model._modules.OSCAR_ERF_wmghg import OSCAR_ERF_wmghg
from oscar._core._model._modules.OSCAR_ERF_slcf import OSCAR_ERF_slcf
from oscar._core._model._modules.OSCAR_ERF_albedo import OSCAR_ERF_albedo

## make module
OSCAR_ERF = Model('OSCAR_ERF')
for model in [OSCAR_ERF_wmghg, OSCAR_ERF_slcf, OSCAR_ERF_albedo]:
    OSCAR_ERF = OSCAR_ERF.merge(model, new_name=OSCAR_ERF.name)


##=====================
## Diagnostic variables
##=====================

## total ERF
OSCAR_ERF.process(
    Out = 'ERF', 
    In = ('ERF_wmghg', 'ERF_swv', 'ERF_O3', 'ERF_aer', 'ERF_lap', 'ERF_luc', 'ERF_aic', 'ERF_volc', 'ERF_solar'), 
    Eq = lambda Var, Par: Eq__ERF(Var, Par), 
    units = 'W m-2')

def Eq__ERF(Var, Par):
    return Var.ERF_wmghg + Var.ERF_swv + Var.ERF_O3 + Var.ERF_aer + Var.ERF_lap + Var.ERF_luc + Var.ERF_aic + Par.e_volc * Var.ERF_volc + Var.ERF_solar


## total ERF occuring in the atmosphere
OSCAR_ERF.process(
    Out = 'ERF_atm', 
    In = ('ERF_CO2', 'ERF_nonCO2', 'ERF_swv', 'ERF_O3', 'ERF_aer', 'ERF_ari_BC', 'ERF_lap', 'ERF_luc', 'ERF_aic', 'ERF_volc', 'ERF_solar'), 
    Eq = lambda Var, Par: Eq__ERF_atm(Var, Par), 
    units = 'W m-2')

def Eq__ERF_atm(Var, Par):
    ERF_atm_CO2 = Par.p_atm_CO2 * Var.ERF_CO2
    ERF_atm_nonCO2 = Par.p_atm_nonCO2 * (Var.ERF_nonCO2 + Var.ERF_swv)
    ERF_atm_O3 = Par.p_atm_O3 * Var.ERF_O3
    ERF_atm_scatter = Par.p_atm_scatter * (Var.ERF_aer - Var.ERF_ari_BC + Par.e_volc * Var.ERF_volc + Var.ERF_aic)
    ERF_atm_absorb = Par.p_atm_absorb * Var.ERF_ari_BC
    ERF_atm_alb = Par.p_atm_alb * (Var.ERF_lap + Var.ERF_luc)
    ERF_atm_solar = Par.p_atm_solar * Var.ERF_solar
    return ERF_atm_CO2 + ERF_atm_nonCO2 + ERF_atm_O3 + ERF_atm_scatter + ERF_atm_absorb + ERF_atm_alb + ERF_atm_solar

