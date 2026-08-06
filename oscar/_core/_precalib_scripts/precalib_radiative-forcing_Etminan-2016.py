import warnings
import numpy as np
import xarray as xr

from oscar._io.paths import get_paths
from oscar._core._base.fct_precalib import run_precalib, make_one_fit, get_best_fit

path_precalib_in = get_paths()["precalib_data"]


name = 'radiative-forcing_Etminan-2016'


##################################################
##   PRECALIBRATION OF SARF WMGHG
##################################################

##=========
## Function
##=========

## precalibration function
def precalib_params(no_warnings=True):
    with warnings.catch_warnings():
        if no_warnings: warnings.filterwarnings('ignore')

        ## get original data
        with xr.open_dataset(path_precalib_in / f'{name}.nc') as TMP:
            ds = TMP.load()

        ## define reference state
        ds_ref = ds.sel(exp=1, drop=True)

        ## initialization
        Par = xr.Dataset()


        ## STEP 1        
        ## each GHG separately under ref state for other GHGs

        ## make model
        def f_SARF_no_overlap(g, aG, bG):
            return aG / bG * (g ** bG - 1)
        
        ## make parameters
        params0 = {'aG': dict(value=0, min=0), 
            'bG': dict(value=1)}

        ## loop on GHGs
        for GHG in ['CO2', 'CH4', 'N2O']:

            ## get experiments under ref state
            if GHG == 'CO2':
                exp_ghg = ds.exp.where((ds.CH4 == ds_ref.CH4) & (ds.N2O == ds_ref.N2O)).dropna('exp', how='all').values
            elif GHG == 'CH4':
                exp_ghg = ds.exp.where((ds.CO2 == ds_ref.CO2) & (ds.N2O == ds_ref.N2O)).dropna('exp', how='all').values
            elif GHG == 'N2O':
                exp_ghg = ds.exp.where((ds.CO2 == ds_ref.CO2) & (ds.CH4 == ds_ref.CH4)).dropna('exp', how='all').values

            ## select data
            xdata = (ds.sel(exp=exp_ghg).drop_vars([var for var in ds if var != GHG]) / ds_ref[GHG].values).rename({GHG: 'g'})
            ydata = ds.sel(exp=exp_ghg)['SARF']
    
            ## fit
            params_fit = make_one_fit(xdata, ydata, f_SARF_no_overlap, params0, 
                file_name=name + f'/sarf_{GHG}')[0].best_values

            ## assign parameters
            Par['Ph_'+GHG] = params_fit['aG']
            Par['Ph_'+GHG].attrs['units'] = 'W m-2'
            Par['x_rf_'+GHG] = params_fit['bG']
            Par['x_rf_'+GHG].attrs['units'] = '1'
            Par[GHG+'_rf'] = ds_ref[GHG].values
            Par[GHG+'_rf'].attrs['units'] = 'ppm' if GHG == 'CO2' else 'ppb'


        ## STEP 2    
        ## all GHGs together with some fixed params

        ## make model
        def f_SARF_overlap(c, m, n, aC, bC, aM, bM, aN, bN, aCN, bCN, aNC, bNC, aMN, bMN, aNM, bNM):
            RF_CO2 = aC / bC * (c**bC - 1)
            RF_CH4 = aM / bM * (m**bM - 1)
            RF_N2O = aN / bN * (n**bN - 1)
            RF_overlap1 = aC * aCN / bC / bCN * (c**bC - 1) * (n**bCN - 1) + aN * aNC / bN / bNC * (c**bNC - 1) * (n**bN - 1)
            RF_overlap2 = aM * aMN / bM / bMN * (m**bM - 1) * (n**bMN - 1) + aN * aNM / bN / bNM * (m**bNM - 1) * (n**bN - 1)
            return RF_CO2 + RF_CH4 + RF_N2O + RF_overlap1 + RF_overlap2

        ## make parameters (params2 to test power law fixed to sqrt)
        params1 = {'aC': dict(value=Par['Ph_CO2'].values, vary=False), 
            'bC': dict(value=Par['x_rf_CO2'].values, vary=False),
            'aM': dict(value=Par['Ph_CH4'].values, vary=False), 
            'bM': dict(value=Par['x_rf_CH4'].values, vary=False),
            'aN': dict(value=Par['Ph_N2O'].values, vary=False), 
            'bN': dict(value=Par['x_rf_N2O'].values, vary=False),
            'aCN': dict(value=0., max=0.), 
            'bCN': dict(value=0.5),  
            'aNC': dict(value=0., max=0.),  
            'bNC': dict(value=0.5, expr='bCN'),   
            'aMN': dict(value=0., max=0.),  
            'bMN': dict(value=0.5),   
            'aNM': dict(value=0., max=0.),  
            'bNM': dict(value=0.5, expr='bMN')}
        params2 = {key: val for key, val in params1.items()}
        params2['bCN'] = params2['bMN'] = dict(value=0.5, vary=False)

        ## make data
        xdata = (ds.drop_vars('SARF') / ds_ref).rename({'CO2': 'c', 'CH4': 'm', 'N2O': 'n'})
        ydata = ds['SARF']
        weights = (1/ydata).where(~np.isinf(1/ydata), 0) # for normalization

        ## fit
        #params_fit = make_one_fit(xdata, ydata, f_SARF_overlap, params1, weights=weights, file_name=name + f'/sarf_GHG')[0].best_values
        params_fit = get_best_fit(xdata, ydata, [f_SARF_overlap], [params1, params2], 
            select_crit='BIC', weights=weights, file_name=name + f'/sarf_GHG')
        
        ## assign parameters
        Par['i_CO2_N2O'] = params_fit['aCN']
        Par['i_N2O_CO2'] = params_fit['aNC']
        Par['i_CH4_N2O'] = params_fit['aMN']
        Par['i_N2O_CH4'] = params_fit['aNM']
        for var in ['i_CO2_N2O', 'i_N2O_CO2','i_CH4_N2O', 'i_N2O_CH4']: 
            Par[var].attrs['units'] = '1'
        Par['x_rf_CO2_N2O'] = params_fit['bCN']
        Par['x_rf_N2O_CO2'] = params_fit['bNC']
        Par['x_rf_CH4_N2O'] = params_fit['bMN']
        Par['x_rf_N2O_CH4'] = params_fit['bNM']
        for var in ['x_rf_CO2_N2O', 'x_rf_N2O_CO2','x_rf_CH4_N2O', 'x_rf_N2O_CH4']: 
            Par[var].attrs['units'] = '1'


    ## return
    return Par.astype(np.float32)


##================
## Generate Params
##================
if __name__ == '__main__':

    ## do precalib
    run_precalib(name, precalib_params, regional=False)

