import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model
from oscar._core._base.fct_solve import safe_exp


#####################################################################
##   NATURAL EMISSIONS
#####################################################################

## initialize
OSCAR_NatEm = Model('OSCAR_NatEm')

## module based on:
## (Thornhill et al., 2021; https://doi.org/10.5194/acp-21-1105-2021)
## note: coarse first-order estimates, since simply linear on global temperature change


##=====================
## Secondary parameters
##=====================

## preindustrial DMS emissions
OSCAR_NatEm.process(
    Out = 'Enat_DMS_pi', 
    Eq = lambda Par: Eq__Enat_DMS_pi(Par), 
    units = 'TgS yr-1')

def Eq__Enat_DMS_pi(Par):
    return Par.Enat_DMS_pd / safe_exp(Par.g_DMS * Par.D_Tg_pd, 5.)


## preindustrial BVOC emissions
OSCAR_NatEm.process(
    Out = 'Enat_BVOC_pi', 
    Eq = lambda Par: Eq__Enat_BVOC_pi(Par), 
    units = 'Tg yr-1')

def Eq__Enat_BVOC_pi(Par):
    fct_lcc = 1 + Par.i_BVOC_Afor * (Par.Aland_pd.sel(bio_land='Forest', drop=True).sum('reg_land', min_count=1) / Par.Aland_pi.sel(bio_land='Forest', drop=True).sum('reg_land', min_count=1) - 1)
    fct_T = safe_exp(Par.g_BVOC * Par.D_Tg_pd, 5.)
    return Par.Enat_BVOC_pd / fct_lcc / fct_T


## present-day LNOx emissions
OSCAR_NatEm.process(
    Out = 'Enat_LNOx_pd', 
    Eq = lambda Par: Eq__Enat_LNOx_pd(Par), 
    units = 'TgN yr-1')

def Eq__Enat_LNOx_pd(Par):
    return Par.Enat_LNOx_pi * safe_exp(Par.g_LNOx * Par.D_Tg_pd, 10.)


## preindustrial SNOx emissions
OSCAR_NatEm.process(
    Out = 'Enat_SNOx_pi', 
    Eq = lambda Par: Eq__Enat_SNOx_pi(Par), 
    units = 'TgN yr-1')

def Eq__Enat_SNOx_pi(Par):
    return Par.Enat_SNOx_pd


##=====================
## Diagnostic variables
##=====================

## mineral dust emissions
OSCAR_NatEm.process(
    Out = 'D_Edust', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_Edust(Var, Par), 
    units = 'Tg yr-1')

def Eq__D_Edust(Var, Par):
    return Par.Edust_pi * (safe_exp(Par.g_dust * Var.D_Tg, 5.) - 1)


## sea salt emissions
OSCAR_NatEm.process(
    Out = 'D_Esalt', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_Esalt(Var, Par), 
    units = 'Tg yr-1')

def Eq__D_Esalt(Var, Par):
    return Par.Esalt_pi * (safe_exp(Par.g_salt * Var.D_Tg, 5.) - 1)


## oceanic DMS emissions
OSCAR_NatEm.process(
    Out = 'D_Enat_DMS', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_Enat_DMS(Var, Par), 
    units = 'TgS yr-1')

def Eq__D_Enat_DMS(Var, Par):
    return Par.Enat_DMS_pi * (safe_exp(Par.g_DMS * Var.D_Tg, 5.) - 1)


## biogenic VOC emissions
OSCAR_NatEm.process(
    Out = 'D_Enat_BVOC', 
    In = ('D_Tg', 'D_Aland'), 
    Eq = lambda Var, Par: Eq__D_Enat_BVOC(Var, Par), 
    units = 'Tg yr-1')

def Eq__D_Enat_BVOC(Var, Par):
    fct_lcc = 1 + Par.i_BVOC_Afor * Var.D_Aland.sel(bio_land='Forest', drop=True).sum('reg_land', min_count=1) / Par.Aland_pi.sel(bio_land='Forest', drop=True).sum('reg_land', min_count=1)
    fct_T = safe_exp(Par.g_BVOC * Var.D_Tg, 5.)
    return Par.Enat_BVOC_pi * (fct_lcc * fct_T - 1)


## lightning NOx emissions
## note: doubled cap of safe exponential
OSCAR_NatEm.process(
    Out = 'D_Enat_LNOx', 
    In = ('D_Tg',), 
    Eq = lambda Var, Par: Eq__D_Enat_LNOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Enat_LNOx(Var, Par):
    return Par.Enat_LNOx_pi * (safe_exp(Par.g_LNOx * Var.D_Tg, 10.) - 1)


## soil NOx emissions
## note: placeholder
OSCAR_NatEm.process(
    Out = 'D_Enat_SNOx', 
    In = (), 
    Eq = lambda Var, Par: Eq__D_Enat_SNOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__D_Enat_SNOx(Var, Par):
    return 0.


## ADDITIONAL DIAGNOSTICS

## total mineral dust emissions
OSCAR_NatEm.process(
    Out = 'Edust', 
    In = ('D_Edust',), 
    Eq = lambda Var, Par: Eq__Edust(Var, Par), 
    units = 'Tg yr-1')

def Eq__Edust(Var, Par):
    return Var.D_Edust + Par.Edust_pi


## total sea salt emissions
OSCAR_NatEm.process(
    Out = 'Esalt', 
    In = ('D_Esalt',), 
    Eq = lambda Var, Par: Eq__Esalt(Var, Par), 
    units = 'Tg yr-1')

def Eq__Esalt(Var, Par):
    return Var.D_Esalt + Par.Esalt_pi


## total oceanic DMS emissions
OSCAR_NatEm.process(
    Out = 'Enat_DMS', 
    In = ('D_Enat_DMS',), 
    Eq = lambda Var, Par: Eq__Enat_DMS(Var, Par), 
    units = 'TgS yr-1')

def Eq__Enat_DMS(Var, Par):
    return Var.D_Enat_DMS + Par.Enat_DMS_pi


## total biogenic VOC emissions
OSCAR_NatEm.process(
    Out = 'Enat_BVOC', 
    In = ('D_Enat_BVOC',), 
    Eq = lambda Var, Par: Eq__Enat_BVOC(Var, Par), 
    units = 'Tg yr-1')

def Eq__Enat_BVOC(Var, Par):
    return Var.D_Enat_BVOC + Par.Enat_BVOC_pi


## total lightning NOx emissions
OSCAR_NatEm.process(
    Out = 'Enat_LNOx', 
    In = ('D_Enat_LNOx',), 
    Eq = lambda Var, Par: Eq__Enat_LNOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__Enat_LNOx(Var, Par):
    return Var.D_Enat_LNOx + Par.Enat_LNOx_pi


## total soil NOx emissions
OSCAR_NatEm.process(
    Out = 'Enat_SNOx', 
    In = ('D_Enat_SNOx',), 
    Eq = lambda Var, Par: Eq__Enat_SNOx(Var, Par), 
    units = 'TgN yr-1')

def Eq__Enat_SNOx(Var, Par):
    return Var.D_Enat_SNOx + Par.Enat_SNOx_pi

