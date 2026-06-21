import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   CLIMATE RESPONSE
#####################################################################

## initialize
OSCAR_clim = Model('OSCAR_clim')


## global temperature: classic two-Layer energy balance model from:
## (Geoffroy et al., 2013; https://doi.org/10.1175/JCLI-D-12-00196.1)

## global precipitation: 
## (Allan et al., 2013; https://doi.org/10.1007/s10712-012-9213-z)


##=====================
## Secondary parameters
##=====================

## equilibrium climate sensitivity
OSCAR_clim.process(
    Out = 'ECS', 
    Eq = lambda Par: Eq__ECS(Par), 
    units = 'K')

def Eq__ECS(Par):
    if 'ERF_CO2_2x' not in Par: return None
    return Par.ERF_CO2_2x / Par.lambda_0


##=====================
## Diagnostic variables
##=====================

## trend in global mean surface temperature
OSCAR_clim.process(
    Out = 'd_Tg', 
    In = ('D_Tg', 'D_Td', 'ERF'), 
    Eq = lambda Var, Par: Eq__d_Tg(Var, Par), 
    units = 'K yr-1')

def Eq__d_Tg(Var, Par):
    return 1/Par.Th_g * (Var.ERF - Par.lambda_0 * Var.D_Tg - Par.e_ohu * Par.th_0 * (Var.D_Tg - Var.D_Td))


## trend in deep ocean temperature
OSCAR_clim.process(
    Out = 'd_Td', 
    In = ('D_Tg', 'D_Td'), 
    Eq = lambda Var, Par: Eq__d_Td(Var, Par), 
    units = 'K yr-1')

def Eq__d_Td(Var, Par):
    return 1/Par.Th_d * Par.th_0 * (Var.D_Tg - Var.D_Td)


## ocean heat uptake
OSCAR_clim.process(
    Out = 'd_OHC', 
    In = ('d_Tg', 'd_Td'), 
    Eq = lambda Var, Par: Eq__d_OHC(Var, Par), 
    units = 'W m-2')

def Eq__d_OHC(Var, Par):
    return Par.p_ohc * (Par.Th_g * Var.d_Tg + Par.Th_d * Var.d_Td)


## ocean heat content
OSCAR_clim.process(
    Out = 'D_OHC', 
    In = ('D_Tg', 'D_Td'), 
    Eq = lambda Var, Par: Eq__D_OHC(Var, Par), 
    units = 'W yr m-2')

def Eq__D_OHC(Var, Par):
    return Par.p_ohc * (Par.Th_g * Var.D_Tg + Par.Th_d * Var.D_Td)


## climate feedback factor
OSCAR_clim.process(
    Out = 'lambda', 
    In = ('D_Tg', 'D_Td'), 
    Eq = lambda Var, Par: Eq__lambda(Var, Par), 
    units = 'W m-2 K-1')

def Eq__lambda(Var, Par):
    return Par.lambda_0 + Par.th_0 * (Par.e_ohu - 1) * (1 - Var.D_Td / Var.D_Tg)


## regional mean temperature
OSCAR_clim.process(
    Out = 'D_Tl', 
    In = ('D_Tg', ), 
    Eq = lambda Var, Par: Eq__D_Tl(Var, Par), 
    units = 'K')

def Eq__D_Tl(Var, Par):
    return Par.a_Tl_Tg * Var.D_Tg


## regional mean precipitation
OSCAR_clim.process(
    Out = 'D_Pl', 
    In = ('D_Pg', 'D_Tl'), 
    Eq = lambda Var, Par: Eq__D_Pl(Var, Par), 
    units = 'mm yr-1')

def Eq__D_Pl(Var, Par):
    return Par.a_Pl_Pg * Var.D_Pg + Par.a_Pl_Tl * Var.D_Tl


##=====================
## Node variables
##=====================

## global mean precipitation
## note: made into a node variable because of ERF of SLCFs from BB emissions is instantaneous
OSCAR_clim.process(
    Out = 'D_Pg', 
    In = ('D_Pg', 'D_Tg', 'ERF_atm'), 
    Eq = lambda Var, Par: Eq__D_Pg(Var, Par), 
    units = 'mm yr-1')

def Eq__D_Pg(Var, Par):
    return Par.a_Pg_Tg * Var.D_Tg + Par.a_Pg_ERF_CO2 / Par.p_atm_CO2 * Var.ERF_atm


##=====================
## Prognostic variables
##=====================

## global mean surface temperature
OSCAR_clim.process(
    Out = 'D_Tg', 
    In = ('D_Tg', 'd_Tg'), 
    DiffEq = lambda Var, Par: DiffEq__D_Tg(Var, Par), 
    vLin = lambda Par: vLin__D_Tg(Par), 
    units = 'K')

def DiffEq__D_Tg(Var, Par):
    return Var.d_Tg

def vLin__D_Tg(Par):
    return (Par.lambda_0 + Par.e_ohu * Par.th_0) / Par.Th_g


## deep ocean temperature
OSCAR_clim.process(
    Out = 'D_Td', 
    In = ('D_Td', 'd_Td'), 
    DiffEq = lambda Var, Par: DiffEq__D_Td(Var, Par), 
    vLin = lambda Par: vLin__D_Td(Par), 
    units = 'K')

def DiffEq__D_Td(Var, Par):
    return Var.d_Td

def vLin__D_Td(Par):
    return Par.th_0 / Par.Th_d

