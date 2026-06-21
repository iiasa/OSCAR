import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   EFFECTIVE RADIATIVE FORCING - SURFACE ALBEDO
#####################################################################

## initialize
OSCAR_ERF_albedo = Model('OSCAR_ERF_albedo')

## module adapted from:
## (Raisanen et al., 2022; https://doi.org/10.5194/acp-22-11579-2022)
## (Ouyang et al., 2022; https://doi.org/10.1038/s41467-022-31558-z)


##=====================
## Diagnostic variables
##=====================

## light-absorbing particles deposition on snow
OSCAR_ERF_albedo.process(
    Out = 'ERF_lap', 
    In = ('D_Eant_BC', 'D_Ebb_BC'), 
    In2 = ('D_Ebb2_BC',),
    Eq = lambda Var, Par: Eq__ERF_lap(Var, Par), 
    units = 'W m-2')

def Eq__ERF_lap(Var, Par):
    ## split BB emissions
    D_Ebb1 = Var.D_Ebb_BC
    D_Ebb2 = Var.D_Ebb2_BC if 'D_Ebb2_BC' in Var else 0. * D_Ebb1
    if 'reg_land' not in D_Ebb1.dims and 'reg_land' in D_Ebb2.dims: 
        D_Ebb1 -= D_Ebb2.sum('reg_land', min_count=1)
    elif 'reg_land' in D_Ebb1.dims and 'reg_land' not in D_Ebb2.dims: 
        D_Ebb1 -= D_Ebb2 * (Par.reg_land == Par.reg_land.isel(reg_land=0, drop=True))
    else:
        D_Ebb1 -= D_Ebb2
    ## apply best model
    if 'reg_land' in Var.D_Eant_BC.dims: IRF_lap_ant = (Par.ph_lap_BC * Var.D_Eant_BC).sum('reg_land', min_count=1)
    else: IRF_lap_ant = Par.ph_lap_BC_glb * Var.D_Eant_BC
    if 'reg_land' in D_Ebb1.dims: IRF_lap_bb1 = (Par.ph_lap_BC * D_Ebb1).sum('reg_land', min_count=1)
    else: IRF_lap_bb1 = Par.ph_lap_BC_glb * D_Ebb1
    if 'reg_land' in D_Ebb2.dims: IRF_lap_bb2 = (Par.ph_lap_BC * D_Ebb2).sum('reg_land', min_count=1)
    else: IRF_lap_bb2 = Par.ph_lap_BC_glb * D_Ebb2
    return Par.a_adj_lap * (IRF_lap_ant + IRF_lap_bb1 + IRF_lap_bb2)


## land cover change (excluding irrigation)
OSCAR_ERF_albedo.process(
    Out = 'ERF_lcc', 
    In = ('D_Aland',), 
    Eq = lambda Var, Par: Eq__ERF_lcc(Var, Par), 
    units = 'W m-2')

def Eq__ERF_lcc(Var, Par):
    return Par.k_ERF_lcc * (Par.ph_lcc * Var.D_Aland).sum('bio_land', min_count=1).sum('reg_land', min_count=1)


## land use change (including irrigation)
OSCAR_ERF_albedo.process(
    Out = 'ERF_luc', 
    In = ('ERF_lcc', 'ERF_irrig'), 
    Eq = lambda Var, Par: Eq__ERF_luc(Var, Par), 
    units = 'W m-2')

def Eq__ERF_luc(Var, Par):
    return Var.ERF_lcc + Var.ERF_irrig

