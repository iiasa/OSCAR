import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib, make_one_fit, get_best_fit
from oscar._core._base.fct_regions import aggreg_regions

path_precalib_in = get_paths()["precalib_data"]


name = 'global-climate_CMIP6'


##################################################
##   PRECALIBRATION OF GLOBAL CLIMATE
##################################################

##==========
## Ancillary
##==========

## parameter conversion from box-model to IRF
## (Geoffroy et al., 2013a; https://doi.org/10.1175/JCLI-D-12-00195.1) (Table 1)
## (Geoffroy et al., 2013b; https://doi.org/10.1175/JCLI-D-12-00196.1) (Section 2a)
def box_2_irf(l0, Cs, Cd, g, e=1.):
    Cd, g = e * Cd, e * g
    b = (l0 + g) / Cs + g / Cd
    bb = (l0 + g) / Cs - g / Cd
    delta = b**2 - 4 * l0 * g / Cs / Cd
    phiF = 0.5 * Cs / g * (bb - np.sqrt(delta))
    phiS = 0.5 * Cs / g * (bb + np.sqrt(delta))
    tF = 0.5 * Cs * Cd / l0 / g * (b - np.sqrt(delta))
    tS = 0.5 * Cs * Cd / l0 / g * (b + np.sqrt(delta))
    aF = tF * l0 / Cs * phiS / (phiS - phiF)
    aS = tS * l0 / Cs * phiF / (phiF - phiS)
    return aS, tS, aF, tF, phiS, phiF


## parameter conversion from IRF to box-model
## (Geoffroy et al., 2013a; https://doi.org/10.1175/JCLI-D-12-00195.1) (Equations 19-21)
## (Geoffroy et al., 2013b; https://doi.org/10.1175/JCLI-D-12-00196.1) (Section 2a)
def irf_2_box(l0, aS, tS, aF, tF, e=1.):
    Cs = l0 / (aF / tF + aS / tS)
    Cd = l0 * (aF * tF + aS * tS) - Cs
    g = Cd / (aS * tF + aF * tS)
    return Cs, Cd/e, g/e


##=========
## Function
##=========

## precalibration function
def precalib_params(no_efficacy=False, no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        with xr.open_dataset(path_precalib_in + '{}.nc'.format(name.replace('global-',''))) as TMP:
            ds = TMP.sel(exp=['piControl', 'abrupt-4xCO2', 'abrupt-2xCO2', 'abrupt-0p5xCO2', '1pctCO2', '1pctCO2-cdr'])
            ds = (ds * ds['_area']).sum('reg_code', min_count=1) / ds['_area'].sum('reg_code', min_count=1)
            ds = ds.drop_vars('_area')
            ds = ds.compute()

        ## offset with preind
        ds_delta = ds.sel(exp=[exp for exp in ds.exp.values if exp != 'piControl']) - ds.sel(exp='piControl', drop=True)

        ## initialization
        Par = xr.Dataset()
        for var in ['F4x', 'lambda_0', 'Th_g', 'Th_d', 'th_0', 'e_ohu', 'a_Pg_Tg', 'a_Pg_ERF_CO2']:
            Par[var] = xr.zeros_like(ds.model, dtype=np.float32)


        ## STEP 1        
        ## temperature

        ## make net TOA radiation model
        ## (Geoffroy et al., 2013b; https://doi.org/10.1175/JCLI-D-12-00196.1) (Equation 3)
        def f_NetRad(tas, H, F0, l0, e):
            return F0 - l0 * tas + (e - 1) * H

        ## make timescales model
        ## (Geoffroy et al., 2013a; https://doi.org/10.1175/JCLI-D-12-00195.1) (Equation 11)
        ## note: differs from paper approach because some models may lead to negative values in log
        def f_Time(t, aF, tF, aS, tS):
            return 1 - aF * np.exp(-t / tF) - aS * np.exp(-t / tS)

        ## func to get deep ocean heat uptake
        def H(t, g, F0, l0, aF, tF, aS, tS):
            return g * F0 / l0 * ((tF / (tF - tS) - aF) * np.exp(-t / tF) + (tS / (tS - tF) - aS) * np.exp(-t / tS))

        ## loop on models
        for mod in ds_delta.model.values:
            print('\n', 'tas', mod, '\n')
            
            ## first guess parameters
            F0, l0, e = 5.35*np.log(4), 5.35*np.log(2)/3, 1.
            aS, aF = 0.5, 0.5
            tS, tF = 200., 4.
            Cs, Cd, g = irf_2_box(l0, aS, tS, aF, tF, e)

            ## loop for iterative fit
            n_loop, max_loop = 0, 1 if no_efficacy else 30
            has_converged = False
            while n_loop < max_loop and not has_converged:
                n_loop += 1
                old_params = [F0, l0, e, Cs, Cd, g]

                ## STEP 1a
                ## net TOA radiation

                ## make parameters
                params_NetRad = {'F0': dict(value=F0, min=0), 
                    'l0': dict(value=l0, min=0),
                    'e': dict(value=e, min=0, vary=n_loop > 1)}

                ## select data
                ds_tmp = ds_delta.drop_vars('pr').sel(exp='abrupt-4xCO2', model=mod).dropna('year', how='any')
                xdata = ds_tmp['tas'].to_dataset(name='tas')
                xdata['H'] = H((ds_tmp.year - ds_tmp.year.values[0] + 0.5), g, F0, l0, aF, tF, aS, tS)
                ydata = sum([mult * ds_tmp[var] for var, mult in zip(['rsdt', 'rsut', 'rlut'], [1, -1, -1])]).to_dataset(name='N')['N']
        
                ## fit
                params_fit = make_one_fit(xdata, ydata, f_NetRad, params_NetRad, 
                    file_name=name + f'/tas_NetRad_{mod}')[0].best_values

                ## update parameters
                F0, l0, e = [params_fit[par] for par in ['F0', 'l0', 'e']]

                ## STEP 1b
                ## timescales
            
                ## make parameters
                params_Time = {'aF': dict(value=aF, min=0, max=1), 
                    'tF': dict(value=tF, min=0),
                    'aS': dict(value=aS, expr='1 - aF'), 
                    'tS': dict(value=tS, min=0)}

                ## select data
                xdata = (ds_tmp.year - ds_tmp.year.values[0] + 0.5).to_dataset(name='t')
                ydata = ds_tmp.rename({'tas': 'T / Teq'})['T / Teq'] * l0 / F0
    
                ## fit
                params_fit = make_one_fit(xdata, ydata, f_Time, params_Time, 
                    file_name=name + f'/tas_Time_{mod}')[0].best_values

                ## update parameters
                aF, tF, aS, tS = [params_fit[par] for par in ['aF', 'tF', 'aS', 'tS']]

                ## get thermal parameters
                Cs, Cd, g = irf_2_box(l0, aS, tS, aF, tF, e)

                ## check convergence
                has_converged = all([np.isclose(old, new) for old, new in zip(old_params, [F0, l0, e, Cs, Cd, g])])
                if has_converged: print(f'* converged in {n_loop} steps *')
                elif n_loop == max_loop: print(f'* not converged in {n_loop} steps *')

            ## assign parameters
            Par['F4x'].loc[{'model': mod}] = F0
            Par['lambda_0'].loc[{'model': mod}] = l0
            Par['Th_g'].loc[{'model': mod}] = Cs
            Par['Th_d'].loc[{'model': mod}] = Cd
            Par['th_0'].loc[{'model': mod}] = g
            Par['e_ohu'].loc[{'model': mod}] = e

        
        ## STEP 2      
        ## precipitation

        ## make linear model
        ## (Allan et al., 2013; https://doi.org/10.1007/s10712-012-9213-z) (Equation 2)
        def f_Precip(tas, F0, a, b):
            return a * tas + b * F0
        
        ## loop on models
        for mod in ds_delta.model.values:
            print('\n', 'pr', mod, '\n')

            ## make parameters
            params_Precip = {'F0': dict(value=Par.F4x.sel(model=mod).values, vary=False), 
                'a': dict(value=0),
                'b': dict(value=0)}

            ## select data
            ds_tmp = ds_delta.drop_vars(['rlut', 'rsut', 'rsdt']).sel(exp='abrupt-4xCO2', model=mod).dropna('year', how='any')
            xdata = ds_tmp['tas'].to_dataset(name='tas')
            ydata = ds_tmp['pr']

            ## fit
            params_fit = make_one_fit(xdata, ydata, f_Precip, params_Precip, 
                file_name=name + f'/pr_Precip_{mod}')[0].best_values
            
            ## assign parameters
            Par['a_Pg_Tg'].loc[{'model': mod}] = params_fit['a']
            Par['a_Pg_ERF_CO2'].loc[{'model': mod}] = params_fit['b']


        ## remove useless parameter and add units
        del Par['F4x']
        Par['lambda_0'].attrs['units'] = 'W m-2 K-1'
        Par['th_0'].attrs['units'] = 'W m-2 K-1'
        Par['Th_g'].attrs['units'] = 'W yr m-2 K-1'
        Par['Th_d'].attrs['units'] = 'W yr m-2 K-1'
        Par['e_ohu'].attrs['units'] = '1'
        Par['a_Pg_Tg'].attrs['units'] = 'mm yr-1 K-1'
        Par['a_Pg_ERF_CO2'].attrs['units'] = 'mm m2 yr-1 W-1'

        ## print remaining NaN for info
        for var in Par:
            if Par[var].isnull().sum() > 0:
                print(f'{var}:', Par[var].isnull().sum().values, 'remaining NaN values!')

        ## make dimensions
        Par = Par.rename({'model': 'mod_clim'})

    ## return
    return Par.astype(np.float32)


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=False)

