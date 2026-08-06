import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import sum_reg


#####################################################################
##   BIOMASS BURNING EMISSIONS
#####################################################################

## initialize
OSCAR_fire = Model('OSCAR_fire')


##=====================
## Secondary parameters
##=====================

## preindustrial biomass burning emissions
OSCAR_fire.process(
    Out = 'Efire_pi', 
    Eq = lambda Par: Eq__Efire_pi(Par), 
    units = 'PgC yr-1')

def Eq__Efire_pi(Par):
    if 'cveg_pi' not in Par: return None
    if 'Cveg_bk_pi' not in Par: return None
    return Par.v_fire * (Par.cveg_pi * Par.Aland_pi + Par.Cveg_bk_pi)


## preindustrial biomass burning CH4 emissions
OSCAR_fire.process(
    Out = 'Ebb_CH4_pi', 
    Eq = lambda Par: Eq__Ebb_CH4_pi(Par), 
    units = 'TgC yr-1')

def Eq__Ebb_CH4_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_CH4 * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_CH4_pi if 'Ebb2_CH4_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning N2O emissions
OSCAR_fire.process(
    Out = 'Ebb_N2O_pi', 
    Eq = lambda Par: Eq__Ebb_N2O_pi(Par), 
    units = 'TgN yr-1')

def Eq__Ebb_N2O_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_N2O * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_N2O_pi if 'Ebb2_N2O_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning H2 emissions
OSCAR_fire.process(
    Out = 'Ebb_H2_pi', 
    Eq = lambda Par: Eq__Ebb_H2_pi(Par), 
    units = 'TgH2 yr-1')

def Eq__Ebb_H2_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_H2 * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_H2_pi if 'Ebb2_H2_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning BC emissions
OSCAR_fire.process(
    Out = 'Ebb_BC_pi', 
    Eq = lambda Par: Eq__Ebb_BC_pi(Par), 
    units = 'TgC yr-1')

def Eq__Ebb_BC_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_BC * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_BC_pi if 'Ebb2_BC_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning OC emissions
OSCAR_fire.process(
    Out = 'Ebb_OC_pi', 
    Eq = lambda Par: Eq__Ebb_OC_pi(Par), 
    units = 'TgC yr-1')

def Eq__Ebb_OC_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_OC * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_OC_pi if 'Ebb2_OC_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning SO2 emissions
OSCAR_fire.process(
    Out = 'Ebb_SO2_pi', 
    Eq = lambda Par: Eq__Ebb_SO2_pi(Par), 
    units = 'TgS yr-1')

def Eq__Ebb_SO2_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_SO2 * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_SO2_pi if 'Ebb2_SO2_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning NH3 emissions
OSCAR_fire.process(
    Out = 'Ebb_NH3_pi', 
    Eq = lambda Par: Eq__Ebb_NH3_pi(Par), 
    units = 'TgN yr-1')

def Eq__Ebb_NH3_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_NH3 * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_NH3_pi if 'Ebb2_NH3_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning NOx emissions
OSCAR_fire.process(
    Out = 'Ebb_NOx_pi', 
    Eq = lambda Par: Eq__Ebb_NOx_pi(Par), 
    units = 'TgN yr-1')

def Eq__Ebb_NOx_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_NOx * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_NOx_pi if 'Ebb2_NOx_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning CO emissions
OSCAR_fire.process(
    Out = 'Ebb_CO_pi', 
    Eq = lambda Par: Eq__Ebb_CO_pi(Par), 
    units = 'TgC yr-1')

def Eq__Ebb_CO_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_CO * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_CO_pi if 'Ebb2_CO_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


## preindustrial biomass burning VOC emissions
OSCAR_fire.process(
    Out = 'Ebb_VOC_pi', 
    Eq = lambda Par: Eq__Ebb_VOC_pi(Par), 
    units = 'Tg yr-1')

def Eq__Ebb_VOC_pi(Par):
    if 'Efire_pi' not in Par: return None
    Ebb1_pi = (Par.a_bb_VOC * Par.Efire_pi).sum('bio_land', min_count=1)
    Ebb2_pi = Par.Ebb2_VOC_pi if 'Ebb2_VOC_pi' in Par else 0. * Ebb1_pi
    if 'reg_land' not in Ebb2_pi.dims: Ebb2_pi = Ebb2_pi * (Ebb1_pi.reg_land == Ebb1_pi.reg_land.isel(reg_land=0, drop=True))
    return Ebb1_pi + Ebb2_pi


##=====================
## Diagnostic variables
##=====================

## fire emissions
OSCAR_fire.process(
    Out = 'D_Efire', 
    In = ('D_efire', 'D_Aland', 'D_Efire_bk'), 
    Eq = lambda Var, Par: Eq__D_Efire(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Efire(Var, Par):
    return Par.v_fire * Par.cveg_pi * Var.D_Aland + Var.D_efire * (Par.Aland_pi + Var.D_Aland) + Var.D_Efire_bk


## biomass burning CH4 emissions
OSCAR_fire.process(
    Out = 'D_Ebb_CH4', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_CH4',),
    Eq = lambda Var, Par: Eq__D_Ebb_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Ebb_CH4(Var, Par):
    D_Ebb1 = (Par.a_bb_CH4 * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_CH4 if 'D_Ebb2_CH4' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning N2O emissions
OSCAR_fire.process(
    Out = 'D_Ebb_N2O', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_N2O',),
    Eq = lambda Var, Par: Eq__D_Ebb_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Ebb_N2O(Var, Par):
    D_Ebb1 = (Par.a_bb_N2O * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_N2O if 'D_Ebb2_N2O' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning H2 emissions
OSCAR_fire.process(
    Out = 'D_Ebb_H2', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_H2',),
    Eq = lambda Var, Par: Eq__D_Ebb_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__D_Ebb_H2(Var, Par):
    D_Ebb1 = (Par.a_bb_H2 * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_H2 if 'D_Ebb2_H2' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning BC emissions
OSCAR_fire.process(
    Out = 'D_Ebb_BC', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_BC',),
    Eq = lambda Var, Par: Eq__D_Ebb_BC(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Ebb_BC(Var, Par):
    D_Ebb1 = (Par.a_bb_BC * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_BC if 'D_Ebb2_BC' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning OC emissions
OSCAR_fire.process(
    Out = 'D_Ebb_OC', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_OC',),
    Eq = lambda Var, Par: Eq__D_Ebb_OC(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Ebb_OC(Var, Par):
    D_Ebb1 = (Par.a_bb_OC * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_OC if 'D_Ebb2_OC' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning SO2 emissions
OSCAR_fire.process(
    Out = 'D_Ebb_SO2', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_SO2',),
    Eq = lambda Var, Par: Eq__D_Ebb_SO2(Var, Par), 
    units = 'TgS yr-1')

def Eq__D_Ebb_SO2(Var, Par):
    D_Ebb1 = (Par.a_bb_SO2 * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_SO2 if 'D_Ebb2_SO2' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning NH3 emissions
OSCAR_fire.process(
    Out = 'D_Ebb_NH3', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_NH3',),
    Eq = lambda Var, Par: Eq__D_Ebb_NH3(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Ebb_NH3(Var, Par):
    D_Ebb1 = (Par.a_bb_NH3 * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_NH3 if 'D_Ebb2_NH3' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning NOx emissions
OSCAR_fire.process(
    Out = 'D_Ebb_NOx', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_NOx',),
    Eq = lambda Var, Par: Eq__D_Ebb_NOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Ebb_NOx(Var, Par):
    D_Ebb1 = (Par.a_bb_NOx * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_NOx if 'D_Ebb2_NOx' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning CO emissions
OSCAR_fire.process(
    Out = 'D_Ebb_CO', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_CO',),
    Eq = lambda Var, Par: Eq__D_Ebb_CO(Var, Par), 
    units = 'TgC yr-1')

def Eq__D_Ebb_CO(Var, Par):
    D_Ebb1 = (Par.a_bb_CO * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_CO if 'D_Ebb2_CO' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## biomass burning VOC emissions
OSCAR_fire.process(
    Out = 'D_Ebb_VOC', 
    In = ('D_Efire',), 
    In2 = ('D_Ebb2_VOC',),
    Eq = lambda Var, Par: Eq__D_Ebb_VOC(Var, Par), 
    units = 'Tg yr-1')

def Eq__D_Ebb_VOC(Var, Par):
    D_Ebb1 = (Par.a_bb_VOC * Var.D_Efire).sum('bio_land', min_count=1)
    D_Ebb2 = Var.D_Ebb2_VOC if 'D_Ebb2_VOC' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb2.dims: D_Ebb2 = D_Ebb2 * (D_Ebb1.reg_land == D_Ebb1.reg_land.isel(reg_land=0, drop=True))
    return D_Ebb1 + D_Ebb2


## ADDITIONAL DIAGNOSTICS

## total biomass burning CH4 emissions
OSCAR_fire.process(
    Out = 'Ebb_CH4', 
    In = ('D_Ebb_CH4',), 
    Eq = lambda Var, Par: Eq__Ebb_CH4(Var, Par), 
    units = 'TgC yr-1')

def Eq__Ebb_CH4(Var, Par):
    return sum_reg(Var.D_Ebb_CH4) + sum_reg(Par.Ebb_CH4_pi)


## total biomass burning N2O emissions
OSCAR_fire.process(
    Out = 'Ebb_N2O', 
    In = ('D_Ebb_N2O',), 
    Eq = lambda Var, Par: Eq__Ebb_N2O(Var, Par), 
    units = 'TgN yr-1')

def Eq__Ebb_N2O(Var, Par):
    return sum_reg(Var.D_Ebb_N2O) + sum_reg(Par.Ebb_N2O_pi)


## total biomass burning H2 emissions
OSCAR_fire.process(
    Out = 'Ebb_H2', 
    In = ('D_Ebb_H2',), 
    Eq = lambda Var, Par: Eq__Ebb_H2(Var, Par), 
    units = 'TgH2 yr-1')

def Eq__Ebb_H2(Var, Par):
    return sum_reg(Var.D_Ebb_H2) + sum_reg(Par.Ebb_H2_pi)


## total biomass burning BC emissions
OSCAR_fire.process(
    Out = 'Ebb_BC', 
    In = ('D_Ebb_BC',), 
    Eq = lambda Var, Par: Eq__Ebb_BC(Var, Par), 
    units = 'TgC yr-1')

def Eq__Ebb_BC(Var, Par):
    return sum_reg(Var.D_Ebb_BC) + sum_reg(Par.Ebb_BC_pi)


## total biomass burning OC emissions
OSCAR_fire.process(
    Out = 'Ebb_OC', 
    In = ('D_Ebb_OC',), 
    Eq = lambda Var, Par: Eq__Ebb_OC(Var, Par), 
    units = 'TgC yr-1')

def Eq__Ebb_OC(Var, Par):
    return sum_reg(Var.D_Ebb_OC) + sum_reg(Par.Ebb_OC_pi)


## total biomass burning SO2 emissions
OSCAR_fire.process(
    Out = 'Ebb_SO2', 
    In = ('D_Ebb_SO2',), 
    Eq = lambda Var, Par: Eq__Ebb_SO2(Var, Par), 
    units = 'TgS yr-1')

def Eq__Ebb_SO2(Var, Par):
    return sum_reg(Var.D_Ebb_SO2) + sum_reg(Par.Ebb_SO2_pi)


## total biomass burning NH3 emissions
OSCAR_fire.process(
    Out = 'Ebb_NH3', 
    In = ('D_Ebb_NH3',), 
    Eq = lambda Var, Par: Eq__Ebb_NH3(Var, Par), 
    units = 'TgN yr-1')

def Eq__Ebb_NH3(Var, Par):
    return sum_reg(Var.D_Ebb_NH3) + sum_reg(Par.Ebb_NH3_pi)


## total biomass burning NOx emissions
OSCAR_fire.process(
    Out = 'Ebb_NOx', 
    In = ('D_Ebb_NOx',), 
    Eq = lambda Var, Par: Eq__Ebb_NOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__Ebb_NOx(Var, Par):
    return sum_reg(Var.D_Ebb_NOx) + sum_reg(Par.Ebb_NOx_pi)


## total biomass burning CO emissions
OSCAR_fire.process(
    Out = 'Ebb_CO', 
    In = ('D_Ebb_CO',), 
    Eq = lambda Var, Par: Eq__Ebb_CO(Var, Par), 
    units = 'TgC yr-1')

def Eq__Ebb_CO(Var, Par):
    return sum_reg(Var.D_Ebb_CO) + sum_reg(Par.Ebb_CO_pi)


## total biomass burning VOC emissions
OSCAR_fire.process(
    Out = 'Ebb_VOC', 
    In = ('D_Ebb_VOC',), 
    Eq = lambda Var, Par: Eq__Ebb_VOC(Var, Par), 
    units = 'Tg yr-1')

def Eq__Ebb_VOC(Var, Par):
    return sum_reg(Var.D_Ebb_VOC) + sum_reg(Par.Ebb_VOC_pi)

