import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   CARBON DIOXIDE BUDGET
#####################################################################

## initialize
OSCAR_CO2 = Model('OSCAR_CO2')


##=====================
## Diagnostic variables
##=====================

## net flux of geological CH4 oxidized in CO2 
## note: unless prescribed, CO2 emissions are assumed to included oxidized CH4 (e.g. because estimated through C content)
## note: does not include D_Emiss_CH4 (which is therefore assumed to be geological)
OSCAR_CO2.process(
    Out = 'D_Foxi_CH4', 
    In = ('D_Eant_CH4', 'D_Ewet_CH4', 'D_Ebb_CH4', 'D_Fsink_CH4'), 
    In2 = ('p_Egeo_CH4',),
    Eq = lambda Var, Par: Eq__D_Foxi_CH4(Var, Par), 
    units='PgC yr-1')

def Eq__D_Foxi_CH4(Var, Par):
    p_Egeo_CH4 = Var.p_Egeo_CH4 if 'p_Egeo_CH4' in Var else 0.
    D_Ebio = (1 - p_Egeo_CH4) * sum_reg(Var.D_Eant_CH4) + sum_reg(Var.D_Ebb_CH4) + Var.D_Ewet_CH4
    D_Foxi = Var.D_Fsink_CH4
    return 1/Par.Pg_to_Tg * (D_Foxi - D_Ebio)


## CO2 total atmospheric sink
OSCAR_CO2.process(
    Out = 'D_Fsink_CO2', 
    In = ('D_Focean', 'D_Fland', 'D_Epf_CO2'), 
    Eq = lambda Var, Par: Eq__D_Fsink_CO2(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fsink_CO2(Var, Par):
    return Var.D_Focean + sum_reg(Var.D_Fland) - Var.D_Epf_CO2


## CO2 missing emissions
## note: zero by default, can be used to prescribe any additional flux
OSCAR_CO2.process(
    Out = 'D_Emiss_CO2', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Emiss_CO2(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Emiss_CO2(Var, Par):
    return 0.


## atmospheric CO2 budget
OSCAR_CO2.process(
    Out = 'd_CO2', 
    In = ('D_Eff', 'D_Eluc', 'D_Foxi_CH4', 'D_Fsink_CO2', 'D_Emiss_CO2'), 
    Eq = lambda Var, Par: Eq__d_CO2(Var, Par),
    units = 'ppm yr-1')

def Eq__d_CO2(Var, Par):
    return 1 / Par.a_CO2 * (sum_reg(Var.D_Eff) + sum_reg(Var.D_Eluc) + Var.D_Foxi_CH4 - Var.D_Fsink_CO2 + Var.D_Emiss_CO2)


## ADDITIONAL DIAGNOSTICS

## CO2 residual emissions (= budget imbalance)
## note: non-zero only if d_CO2 is prescribed!
OSCAR_CO2.process(
    Out = 'Eimb_CO2', 
    In = ('d_CO2', 'D_Eff', 'D_Eluc', 'D_Foxi_CH4', 'D_Fsink_CO2', 'D_Emiss_CO2'), 
    Eq = lambda Var, Par: Eq__Eimb_CO2(Var, Par), 
    units = 'PgC yr-1')

def Eq__Eimb_CO2(Var, Par):
    return Par.a_CO2 * Var.d_CO2 - (sum_reg(Var.D_Eff) + sum_reg(Var.D_Eluc) - Var.D_Fsink_CO2 + Var.D_Emiss_CO2)


## airborne fraction
OSCAR_CO2.process(
    Out = 'AF', 
    In = ('d_CO2', 'D_Eff', 'D_Eluc'), 
    Eq = lambda Var, Par: Eq__AF(Var, Par), 
    units = '1')

def Eq__AF(Var, Par):
    return Par.a_CO2 * Var.d_CO2 / (sum_reg(Var.D_Eff) + sum_reg(Var.D_Eluc))


## carbon sinks rate
OSCAR_CO2.process(
    Out = 'kS', 
    In = ('D_CO2', 'D_Fsink_CO2'), 
    Eq = lambda Var, Par: Eq__kS(Var, Par), 
    units = 'yr-1')

def Eq__kS(Var, Par):
    return Var.D_Fsink_CO2 / Par.a_CO2 / Var.D_CO2


## total atmospheric CO2 concentration
OSCAR_CO2.process(
    Out = 'CO2', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__CO2(Var, Par), 
    units = 'ppm')

def Eq__CO2(Var, Par):
    return Par.CO2_pi + Var.D_CO2


##=====================
## Prognostic variables
##=====================

## atmospheric CO2 concentration
OSCAR_CO2.process(
    Out = 'D_CO2', 
    In = ('D_CO2', 'd_CO2'), 
    DiffEq = lambda Var, Par: DiffEq__D_CO2(Var, Par), 
    units = 'ppm')

def DiffEq__D_CO2(Var, Par):
    return Var.d_CO2

