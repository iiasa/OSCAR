import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   LAND CARBON CYCLE
#####################################################################

## import submodules
from oscar._core._model._modules.OSCAR_landC_density import OSCAR_landC_density
from oscar._core._model._modules.OSCAR_landC_bk import OSCAR_landC_bk

## make module
OSCAR_landC = Model('OSCAR_landC')
for model in [OSCAR_landC_density, OSCAR_landC_bk]:
    OSCAR_landC = OSCAR_landC.merge(model, new_name=OSCAR_landC.name)


##=====================
## Diagnostic variables
##=====================

## COMBINED VARIABLES

## land-use change emissions
OSCAR_landC.process(
    Out = 'D_Eluc', 
    In = ('D_NBP_bk',), 
    Eq = lambda Var, Par: Eq__D_Eluc(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Eluc(Var, Par):
    #D_NBP_luc = Var.D_NBP_bk.sel(dist_bk=Par.dim_luc.values).sum('dist_bk', min_count=1)
    return -Var.D_NBP_bk.sum('bio_land', min_count=1)


## land carbon sink (excluding permafrost)
OSCAR_landC.process(
    Out = 'D_Fland', 
    In = ('D_nbp', 'D_Aland'), 
    Eq = lambda Var, Par: Eq__D_Fland(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fland(Var, Par):
    return (Var.D_nbp * (Par.Aland_pi + Var.D_Aland)).sum('bio_land', min_count=1)


## land carbon sink (including permafrost)
OSCAR_landC.process(
    Out = 'D_Fland_pf', 
    In = ('D_Fland',), 
    In2 = ('D_Epf_CO2',),
    Eq = lambda Var, Par: Eq__D_Fland_pf(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fland_pf(Var, Par):
    return Var.D_Fland.sum('reg_land', min_count=1) - (Var.D_Epf_CO2 if 'D_Epf_CO2' in Var else 0.)


## replaced sources and sinks (RSS) 
## note: this is the loss of sink capacity (LASC) according to
## (Gasser & Ciais, 2013; https://doi.org/10.5194/esd-4-171-2013)
OSCAR_landC.process(
    Out = 'D_Frss', 
    In = ('D_nbp', 'D_Aland'), 
    Eq = lambda Var, Par: Eq__D_Frss(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Frss(Var, Par):
    return (Var.D_nbp * Var.D_Aland).sum('bio_land', min_count=1)


## ADDITIONAL DIAGNOSTICS

## total NBP (i.e. net atmosphere-to-land flux)
OSCAR_landC.process(
    Out = 'NBP', 
    In = ('D_nbp', 'D_Aland', 'D_NBP_bk'), 
    Eq = lambda Var, Par: Eq__NPP(Var, Par), 
    units = 'PgC yr-1')

def Eq__NBP(Var, Par):
    NBP_env = Var.D_nbp * (Par.Aland_pi + Var.D_Aland)
    NBP_bk = Var.D_NBP_bk #.sum('dist_bk', min_count=1)
    return NBP_env + NBP_bk


## total NPP
OSCAR_landC.process(
    Out = 'NPP', 
    In = ('D_npp', 'D_Aland', 'D_NPP_bk'), 
    Eq = lambda Var, Par: Eq__NPP(Var, Par), 
    units = 'PgC yr-1')

def Eq__NPP(Var, Par):
    NPP_env = (Par.npp_pi + Var.D_npp) * (Par.Aland_pi + Var.D_Aland)
    NPP_bk = Var.D_NPP_bk #.sum('dist_bk', min_count=1)
    return NPP_env + NPP_bk


## total apparent NPP (after extraction)
OSCAR_landC.process(
    Out = 'NPPa', 
    In = ('D_npp', 'D_echarv', 'D_egraz', 'D_Aland', 'D_NPP_bk', 'D_Echarv_bk', 'D_Egraz_bk'), 
    Eq = lambda Var, Par: Eq__NPPa(Var, Par), 
    units = 'PgC yr-1')

def Eq__NPPa(Var, Par):
    NPP_env = (Par.npp_pi * (1 - Par.p_charv - Par.p_graz) + Var.D_npp - Var.D_echarv - Var.D_egraz) * (Par.Aland_pi + Var.D_Aland)
    NPP_bk = Var.D_NPP_bk - Var.D_Echarv_bk - Var.D_Egraz_bk
    return NPP_env + NPP_bk


## total fire emissions
OSCAR_landC.process(
    Out = 'Efire', 
    In = ('D_efire', 'D_Aland', 'D_Efire_bk'), 
    Eq = lambda Var, Par: Eq__Efire(Var, Par), 
    units = 'PgC yr-1')

def Eq__Efire(Var, Par):
    Efire_env = (Par.v_fire * Par.cveg_pi + Var.D_efire) * (Par.Aland_pi + Var.D_Aland)
    Efire_bk = Par.v_fire * Par.Cveg_bk_pi + Var.D_Efire_bk
    return Efire_env + Efire_bk


## total heterotrophic respiration
OSCAR_landC.process(
    Out = 'Eresp', 
    In = ('D_ecwd', 'D_esoil', 'D_Aland', 'D_Ecwd_bk', 'D_Esoil_bk'), 
    Eq = lambda Var, Par: Eq__Eresp(Var, Par), 
    units = 'PgC yr-1')

def Eq__Eresp(Var, Par):
    Eresp_env = (Par.p_cwd_resp * Par.v_cwd * Par.ccwd_pi + Var.D_ecwd + Par.v_resp * Par.csoil_pi + Var.D_esoil) * (Par.Aland_pi + Var.D_Aland)
    Eresp_bk = (Par.p_cwd_resp * Par.v_cwd * Par.Ccwd_bk_pi + Var.D_Ecwd_bk + Par.v_resp * Par.Csoil_bk_pi + Var.D_Esoil_bk)
    return Eresp_env + Eresp_bk


## total harvested wood products decay
OSCAR_landC.process(
    Out = 'Ehwp', 
    In = ('D_Ehwp_bk',), 
    Eq = lambda Var, Par: Eq__Ehwp(Var, Par), 
    units = 'PgC yr-1')

def Eq__Ehwp(Var, Par):
    return (Par.v_hwp * Par.Chwp_bk_pi +  Var.D_Ehwp_bk).sum('box_hwp', min_count=1)


## total vegetation carbon stock
OSCAR_landC.process(
    Out = 'Cveg', 
    In = ('D_cveg', 'D_Aland', 'D_Cveg_bk'), 
    Eq = lambda Var, Par: Eq__Cveg(Var, Par), 
    units = 'PgC')

def Eq__Cveg(Var, Par):
    Cveg_env = (Par.cveg_pi + Var.D_cveg) * (Par.Aland_pi + Var.D_Aland)
    Cveg_bk = (Par.Cveg_bk_pi + Var.D_Cveg_bk)
    return Cveg_env + Cveg_bk


## total coarse woody debris carbon stock
OSCAR_landC.process(
    Out = 'Ccwd', 
    In = ('D_ccwd', 'D_Aland', 'D_Ccwd_bk'), 
    Eq = lambda Var, Par: Eq__Ccwd(Var, Par), 
    units = 'PgC')

def Eq__Ccwd(Var, Par):
    Ccwd_env = (Par.ccwd_pi + Var.D_ccwd) * (Par.Aland_pi + Var.D_Aland)
    Ccwd_bk = (Par.Ccwd_bk_pi + Var.D_Ccwd_bk)
    return Ccwd_env + Ccwd_bk


## total soil carbon stock
OSCAR_landC.process(
    Out = 'Csoil', 
    In = ('D_csoil', 'D_Aland', 'D_Csoil_bk'), 
    Eq = lambda Var, Par: Eq__Csoil(Var, Par), 
    units = 'PgC')

def Eq__Csoil(Var, Par):
    Csoil_env = (Par.csoil_pi + Var.D_csoil) * (Par.Aland_pi + Var.D_Aland)
    Csoil_bk = (Par.Csoil_bk_pi + Var.D_Csoil_bk)
    return Csoil_env + Csoil_bk


## total harvested wood product stock
OSCAR_landC.process(
    Out = 'Chwp', 
    In = ('D_Chwp_bk',), 
    Eq = lambda Var, Par: Eq__Chwp(Var, Par), 
    units = 'PgC')

def Eq__Chwp(Var, Par):
    return (Par.Chwp_bk_pi + Var.D_Chwp_bk).sum('box_hwp', min_count=1)


## total land carbon stock (excl. permafrost!)
OSCAR_landC.process(
    Out = 'Cland', 
    In = ('Cveg', 'Ccwd', 'Csoil', 'Chwp'), 
    Eq = lambda Var, Par: Eq__Cland(Var, Par), 
    units = 'PgC')

def Eq__Cland(Var, Par):
    return Var.Cveg + Var.Ccwd + Var.Csoil + Var.Chwp


## actual biome area
OSCAR_landC.process(
    Out = 'Aland', 
    In = ('D_Aland',), 
    Eq = lambda Var, Par: Eq__Aland(Var, Par), 
    units = 'Mha')

def Eq__Aland(Var, Par):
    return Var.D_Aland + Par.Aland_pi


##=====================
## Prognostic variables
##=====================

## biome area
OSCAR_landC.process(
    Out = 'D_Aland', 
    In = ('D_Aland', 'D_dA_lcc1', 'D_dA_lcc2'), 
    DiffEq = lambda Var, Par: DiffEq__D_Aland(Var, Par),
    units = 'Mha', 
    core_dims = ['reg_land', 'bio_land'])

def DiffEq__D_Aland(Var, Par):
    d_Aland_gain = (Var.D_dA_lcc1 + Var.D_dA_lcc2).sum('bio_from', min_count=1).rename({'bio_to':'bio_land'})
    d_Aland_loss = (Var.D_dA_lcc1 + Var.D_dA_lcc2).sum('bio_to', min_count=1).rename({'bio_from':'bio_land'})
    return d_Aland_gain - d_Aland_loss

