import numpy as np
import xarray as xr

from oscar._core._base.cls_main import Model


#####################################################################
##   OCEAN CARBON CYCLE
#####################################################################

## initialize
OSCAR_oceanC = Model('OSCAR_oceanC')

## module adapted from:
## (Joos et al., 1996; https://doi.org/10.3402/tellusb.v48i3.15921)


##=====================
## Secondary parameters
##=====================

## fraction of the flux bypassing the surface ocean
OSCAR_oceanC.process(
    Out = 'p_deep', 
    Eq = lambda Par: Eq__p_deep(Par), 
    units='1')

def Eq__p_deep(Par):
    return 1 - Par.p_surf.sum('box_surf', min_count=1)


##=====================
## Diagnostic variables
##=====================

## partial pressure of CO2 at sea surface
## (Joos et al., 2001; https://doi.org/10.1029/2000GB001375)
OSCAR_oceanC.process(
    Out = 'D_pCO2', 
    In = ('D_dic', 'D_Tg'), 
    Eq = lambda Var, Par: Eq__D_pCO2(Var, Par), 
    units = 'ppm')

def Eq__D_pCO2(Var, Par):
    D_pCO2 = ((1.5568 - 1.3993E-2 * Par.To_0) * Var.D_dic
            + (7.4706 - 0.20207 * Par.To_0) * 1E-3 * Var.D_dic ** 2
            - (1.2748 - 0.12015 * Par.To_0) * 1E-5 * Var.D_dic ** 3
            + (2.4491 - 0.12639 * Par.To_0) * 1E-7 * Var.D_dic ** 4
            - (1.5468 - 0.15326 * Par.To_0) * 1E-10 * Var.D_dic ** 5)
    D_pCO2 = (Par.CO2_pi + Par.k_b_dic * D_pCO2) * np.exp(0.0423 * Par.k_g_dic * Var.D_Tg) - Par.CO2_pi
    return D_pCO2


## dissolved inorganic carbon in the surface layer
OSCAR_oceanC.process(
    Out = 'D_dic', 
    In = ('D_Csurf',), 
    Eq = lambda Var, Par: Eq__D_dic(Var, Par), 
    units = 'umol kgSW-1')

def Eq__D_dic(Var, Par):
    return Par.a_dic / Par.V_mld * Var.D_Csurf.sum('box_surf', min_count=1)


## ingoing flux to the ocean
OSCAR_oceanC.process(
    Out = 'D_Fin', 
    In = ('D_CO2',), 
    Eq = lambda Var, Par: Eq__D_Fin(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fin(Var, Par):
    return Par.v_fg * Par.a_CO2 * Var.D_CO2


## outgoing flux from the ocean
OSCAR_oceanC.process(
    Out = 'D_Fout', 
    In = ('D_pCO2',), 
    Eq = lambda Var, Par: Eq__D_Fout(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Fout(Var, Par):
    return Par.v_fg * Par.a_CO2 * Var.D_pCO2


## circulation flux from surface to deep ocean
OSCAR_oceanC.process(
    Out = 'D_Fcirc', 
    In = ('D_Csurf',), 
    Eq = lambda Var, Par: Eq__D_Fcirc(Var, Par), 
    units = 'PgC yr-1', 
    core_dims = ['box_surf'])

def Eq__D_Fcirc(Var, Par):
    return 1/Par.t_circ * Var.D_Csurf


## ocean carbon sink
OSCAR_oceanC.process(
    Out = 'D_Focean',
    In = ('D_Fin', 'D_Fout'),
    Eq = lambda Var, Par: Eq__D_Focean(Var, Par), 
    units = 'PgC yr-1')

def Eq__D_Focean(Var, Par):
    return Var.D_Fin - Var.D_Fout


## ADDITIONAL DIAGNOSTICS

## total ocean carbon stock
OSCAR_oceanC.process(
    Out = 'D_Cocean',
    In = ('D_Csurf', 'D_Cdeep'),
    Eq = lambda Var, Par: Eq__D_Cocean(Var, Par), 
    units = 'PgC')

def Eq__D_Cocean(Var, Par):
    return Var.D_Csurf.sum('box_surf', min_count=1) + Var.D_Cdeep


##=====================
## Prognostic variables
##=====================

## surface ocean carbon stock
OSCAR_oceanC.process(
    Out = 'D_Csurf', 
    In = ('D_Csurf', 'D_Fcirc', 'D_Fout', 'D_Fin'), 
    DiffEq = lambda Var, Par: DiffEq__D_Csurf(Var, Par), 
    vLin = lambda Par: vLin__D_Csurf(Par), 
    units = 'PgC', 
    core_dims = ['box_surf'])

def DiffEq__D_Csurf(Var, Par):
    return Par.p_surf * (Var.D_Fin - Var.D_Fout) - Var.D_Fcirc

def vLin__D_Csurf(Par):
    return 1 / Par.t_circ


## surface ocean carbon stock
OSCAR_oceanC.process(
    Out = 'D_Cdeep', 
    In = ('D_Cdeep', 'D_Fcirc', 'D_Fout', 'D_Fin'), 
    DiffEq = lambda Var, Par: DiffEq__D_Cdeep(Var, Par), 
    units = 'PgC')

def DiffEq__D_Cdeep(Var, Par):
    return Par.p_deep * (Var.D_Fin - Var.D_Fout) + Var.D_Fcirc.sum('box_surf', min_count=1)

