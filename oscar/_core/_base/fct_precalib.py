import os
import gc
import time
import inspect
import warnings
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

from lmfit import Model

from oscar._io.paths import get_paths, REGIONS_INFO_PATH

path_precalib_out = get_paths()["params_precalib"]
path_precalib_plot = get_paths()["precalib_plots"]
path_reg = REGIONS_INFO_PATH


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## fit metrics
def R2(ydata, ypred):
    return 1 - np.mean((ydata - ypred)**2) / np.mean((ydata - np.mean(ydata))**2)

def MAE(ydata, ypred):
    return np.mean(np.abs(ydata - ypred))

def RMSE(ydata, ypred):
    return np.sqrt(np.mean((ydata - ypred)**2))

def nRMSE(ydata, ypred):
    return np.sqrt(np.mean((ydata - ypred)**2)) / np.mean(ydata)

def BIC0(ydata):
    return len(ydata) * np.log(np.mean((ydata - 0.0)**2))

def BIC1(ydata):
    return len(ydata) * np.log(np.mean((ydata - 1.0)**2))


## extended fit report with extra metrics
def extended_fit_report(result, extra_metrics={}, func=None):
    ## default report
    string = result.fit_report()
    ## replace model with equation
    if func is not None:
        lines = string.split('\n')
        new_lines = [inspect.getsource(func)[:-1] if n==1 else line for n, line in enumerate(lines)]
        string = '\n'.join(new_lines)
    ## add cross validation
    string += '\n[[Cross Validation]] (out-of-sample)'
    for key in extra_metrics:
        string += '\n    {:<10} = {:10f}'.format(key, extra_metrics[key])
    return string


##################################################
##   2. WRAPPER FOR LMFIT
##################################################

## wrapper for one lmfit
def make_one_fit(xdata, ydata, yfunc, params, 
    weights=None, nan_policy='raise',
    time_axis='year', time_len=5, exp_axis='exp', exp_out=[], 
    file_name='', print_report=True):
    print(50*'-', sep='')

    ## handle cases without experiments
    if exp_axis not in xdata.dims:
        exp_out = []
        xdata = xdata.assign_coords(exp='').expand_dims('exp', -1)
        ydata = ydata.assign_coords(exp='').expand_dims('exp', -1)

    ## get data info and merge for common processing
    ydata_name = ydata.name if ydata.name is not None else 'y'
    data = xr.merge([xdata, ydata.to_dataset(name=ydata_name)], join='outer', compat='no_conflicts')

    ## cleanup nan values to get common points
    data_clean = {}
    for exp in data[exp_axis].values:
        data_clean[exp] = data.sel({exp_axis: exp}, drop=True)
        if time_axis in data_clean[exp].dims: 
            data_clean[exp] = data_clean[exp].dropna(time_axis, how='any')

    ## make in-sample and out-of-sample data
    xdata_in = {var: np.hstack([data_clean[exp][var].values for exp in data[exp_axis].values if exp not in exp_out]) for var in xdata}
    ydata_in = np.hstack([data_clean[exp][ydata_name].values for exp in data[exp_axis].values if exp not in exp_out])
    if len(exp_out) > 0:
        ydata_out = np.hstack([data_clean[exp][ydata_name].values for exp in data[exp_axis].values if exp in exp_out])
    else:
        ydata_out = np.array([])

    ## make model & params
    model = Model(yfunc, independent_vars=xdata.keys())
    model_params = model.make_params(**{name: {k: v for k, v in cfg.items() if k != 'default'} for name, cfg in params.items()})

    ## fit
    result = model.fit(ydata_in, model_params, **{var: xdata_in[var] for var in xdata.keys()}, weights=weights, nan_policy=nan_policy)

    ## predicted values
    ## model runs
    ypred = xr.full_like(ydata, np.nan)
    for exp in ypred[exp_axis].values:
        ypred.loc[{exp_axis: exp}] = model.eval(result.params, **{var: xdata[var].sel({exp_axis: exp}) for var in xdata.keys()})
    
    ## merge data and clean up
    data2 = xr.merge([xdata, ypred.to_dataset(name=ydata_name)], join='outer', compat='no_conflicts')
    data2_clean = {}
    for exp in data2[exp_axis].values:
        data2_clean[exp] = data2.sel({exp_axis: exp}, drop=True)
        if time_axis in data2_clean[exp].dims:
            data2_clean[exp] = data2_clean[exp].dropna(time_axis, how='any')
    
    ## sort in and out
    ypred_in = np.hstack([data2_clean[exp][ydata_name].values for exp in data2[exp_axis].values if exp not in exp_out])
    if len(exp_out) > 0:
        ypred_out = np.hstack([data2_clean[exp][ydata_name].values for exp in data2[exp_axis].values if exp in exp_out])
    else:
        ypred_out = np.array([])

    ## get fit metrics
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        metrics = {'MAE_in': MAE(ydata_in, ypred_in),
            'MAE_out': MAE(ydata_out, ypred_out),
            'RMSE_in': RMSE(ydata_in, ypred_in),
            'RMSE_out': RMSE(ydata_out, ypred_out),
            'nRMSE_in': nRMSE(ydata_in, ypred_in),
            'nRMSE_out': nRMSE(ydata_out, ypred_out)}

    ## print fit report
    full_report = extended_fit_report(result, metrics, yfunc)
    if print_report:
        print(full_report)

    ## make control plot
    if len(file_name) > 0:
        plot_fit(xdata, ydata, ypred, full_report,    
            time_axis, time_len, exp_axis, exp_out, 
            file_name)

    ## return
    print(50*'-', sep='')
    return result, metrics


## function to do multiple lmfit and select best
def get_best_fit(xdata, ydata, yfunc_list, params_list, 
    select_crit='R2', fallback=True, test_BIC0=False, test_BIC1=False, 
    keep_all_plots=False, **fit_kwargs):
    
    ## metrics list and check
    metrics_list = ['R2', 'AIC', 'BIC', 'MAE_in', 'MAE_out', 'RMSE_in', 'RMSE_out', 'nRMSE_in', 'nRMSE_out', 'chi2', 'rchi2']
    assert select_crit in metrics_list

    ## make/check lists of yfunc and params
    params0 = (len(params_list) == 1)
    if len(yfunc_list) == 1 and len(params_list) > 1:
        yfunc_list = [yfunc_list[0] for _ in range(len(params_list))]
    elif len(yfunc_list) > 1 and len(params_list) == 1:
        params_list = [params_list[0] for _ in range(len(yfunc_list))]
    elif len(yfunc_list) != len(params_list):
        raise AssertionError(f'length of yfunc_list ({len(yfunc_list)}) does not match length of params_list ({len(params_list)})')

    ## loop on fits
    all_metrics, all_values = [], []
    print('\n', 50*'=', sep='')
    for n, yfunc, params in zip(range(len(yfunc_list)), yfunc_list, params_list):
        print(f'fitting {n+1}/{len(yfunc_list)}:', yfunc.__name__, 'params0' if params0 else f'params{n+1}')

        ## change file name if multiple fits
        file_name = fit_kwargs.get('file_name', '')
        if len(yfunc_list) > 1: file_name = fit_kwargs['file_name'] + f'_fit{n+1}'

        ## make fit
        result, metrics = make_one_fit(xdata, ydata, yfunc, params, **{**fit_kwargs, **{'file_name': file_name}})

        ## check success and replace if not
        if result.success:
            values = result.best_values
            metrics['R2'] = result.rsquared
            metrics['AIC'] = result.aic
            metrics['BIC'] = result.bic
            metrics['chi2'] = result.chisqr
            metrics['rchi2'] = result.redchi
        else:
            values = {key: np.nan for key in result.best_values.keys()}
            for var in metrics_list:
                if var == 'R2': metrics[var] = -np.inf
                else: metrics[var] = np.inf

        ## falls back to default values if bad fit (and allowed)
        fail_R2 = metrics['R2'] < 0.01
        fail_BIC0 = metrics['BIC'] > BIC0(ydata) and test_BIC0
        fail_BIC1 = metrics['BIC'] > BIC1(ydata) and test_BIC1
        if (fail_R2 or fail_BIC0 or fail_BIC1) and fallback:
            print(f'fit {n+1}:', 'falling back to default params')
            print(50*'-', sep='')
            for var in params:
                if 'default' in params[var]:
                    values[var] = params[var]['default']
                else:
                    values[var] = np.nan
                if 'vary' in params[var]:
                    if not params[var]['vary']: 
                        values[var] = params[var]['value']

        ## append outcome
        all_metrics.append(metrics)
        all_values.append(values)

        ## clean up memory
        del result
        gc.collect()

    ## return fit if only one
    if len(yfunc_list) == 1:
        print(50*'=', '\n', sep='')
        return all_values[0]

    ## otherwise do selection
    else:

        ## get selected metric
        metric_eval = np.array([metrics[select_crit] for metrics in all_metrics])

        ## find best value
        if select_crit in ['R2']: n_sel = np.argmax(metric_eval)
        else: n_sel = np.argmin(metric_eval)

        ## print selection
        print(f'selected fit {n_sel+1}:', yfunc_list[n_sel].__name__, 'params0' if params0 else f'params{n_sel+1}')
        print(50*'=', '\n', sep='')
        
        ## clean up figures
        if not keep_all_plots and 'file_name' != '':
            file_name = fit_kwargs['file_name']
            for n in [n for n in range(len(yfunc_list)) if n != n_sel]:
                time.sleep(0.1)
                os.remove(path_precalib_plot / f'{file_name}_fit{n+1}.png')
            if os.path.exists(path_precalib_plot / f'{file_name}.png'):
                time.sleep(0.1)
                os.remove(path_precalib_plot / f'{file_name}.png')
            time.sleep(0.1)
            os.rename(path_precalib_plot / f'{file_name}_fit{n_sel+1}.png', path_precalib_plot / f'{file_name}.png')

        ## return selection
        return all_values[n_sel]


## control plot for fit (rough for now!)
def plot_fit(xdata, ydata, ypred, full_report, 
    time_axis, time_len, exp_axis, exp_out, 
    file_name):

    ## internal parameters for plots
    n_sub = 4
    n_cols = 6

    ## choose plot style
    style_dflt = 'tableau-colorblind10' 
    if style_dflt in plt.style.available:
        plt.style.use(style_dflt)
    elif any(['colorblind' in style for style in plt.style.available]):
        plt.style.use([style for style in plt.style.available][0])

    ## create figure
    fig = plt.figure(figsize=(8, 4))
    fig.add_gridspec(1, n_sub, width_ratios=(n_sub-1)*[n_sub-1]+[1])
    ax = plt.subplot2grid((1, n_sub), (0, 0), colspan=n_sub-1)

    ## loop on experiments
    for exp in xdata[exp_axis].values:
        
        ## choose data    
        if len(xdata) == 1 and time_axis in ydata.dims:
            var = [var for var in xdata][0]
            xx = xdata[var].sel(exp=exp)
            xx_smooth = xdata[var].sel(exp=exp).rolling({time_axis: time_len}, center=True).mean(time_axis)
            yy_smooth = ydata.sel(exp=exp).rolling({time_axis: time_len}, center=True).mean(time_axis)
        elif len(xdata) == 1:
            var = [var for var in xdata][0]
            xx = xdata[var].sel(exp=exp)            
        elif time_axis in ydata.dims:
            xx = ydata[time_axis]
            xx_smooth = ydata[time_axis].rolling({time_axis: time_len}, center=True).mean(time_axis)
            yy_smooth = ydata.sel(exp=exp).rolling({time_axis: time_len}, center=True).mean(time_axis)
        else:
            xx = 1 + np.where(ydata.exp == exp)[0]

        ## plot reference data
        line, = plt.plot(xx, ydata.sel(exp=exp), marker='+', ls='none', ms=4, alpha=0.8, label=f'{exp} ' + ('(out)' if exp in exp_out else '(in)'))
        if time_axis in ydata.dims:
            plt.plot(xx_smooth, yy_smooth, lw=1, ls='-', alpha=0.8, color=line.get_color())
        
        ## plot fitted data
        if time_axis in ypred.dims:
            plt.plot(xx, ypred.sel(exp=exp), color='k', lw=1.5, alpha=0.8, ls='--' if exp in exp_out else '-')
        else:
            plt.plot(xx, ypred.sel(exp=exp), marker='o', mec='k', mfc='none', ms=3, mew=0.5, lw=1.5, alpha=0.8, ls='--' if exp in exp_out else '-')

    ## legend and formatting
    plt.xticks(fontsize='x-small')
    plt.yticks(fontsize='x-small')
    plt.legend(loc=0, ncol=int(np.ceil(len(xdata[exp_axis])/n_cols)), frameon=False, 
        handlelength=0.2, handletextpad=0.5, borderpad=0.2, labelspacing=0.2, fontsize='xx-small')
    if len(xdata) == 1: plt.xlabel(var, fontsize='x-small')
    plt.ylabel(ydata.name, fontsize='x-small')
    plt.title(file_name.split('/')[-1], fontsize='x-small')

    ## add report to figure
    ax = plt.subplot2grid((1, n_sub), (0, n_sub-1))
    ax.axis('off')
    txt = ax.text(0, 1, full_report, ha='left', va='top', fontsize=3+int(4*24/len(full_report.split('\n'))), wrap=True, transform=ax.transAxes)  
    txt.set_in_layout(False)
    
    ## layout and save
    plt.tight_layout()
    if len(file_name) > 0:
        if len(file_name.split('/')) > 1: 
            (path_precalib_plot / file_name).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_precalib_plot / f'{file_name}.png', dpi=200)
        plt.close(fig)
    else:
        plt.show()
    del fig


##################################################
##   3. WRAPPER FOR PRECALIB CALL
##################################################

## simple wrapper
def run_precalib(name, precalib_func, regional=False, mod_region_list=None, **kwargs):
    
    ## print
    print('\n', 50*'=', sep='')
    print('precalibration of params from:', name)
    print(50*'=', '\n', sep='')

    ## if regional parameters
    if regional:

        ## get available mod_region
        if mod_region_list is None:
            with open(path_reg / 'OSCAR_reg_dict.csv') as f: 
                first_line = f.readline()
                mod_region_list = first_line.strip('\n').split(',')[3:]

        ## loop on mod_region
        for mod_region in mod_region_list:
            print('\n', 50*'-', sep='')
            print('mod_region:', mod_region)

            ## generate parameters
            print('precalibrating')
            Par = precalib_func(mod_region, **kwargs)

            ## save output
            print('saving')
            Par.to_netcdf(path_precalib_out / f'{name}__{mod_region}.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Par})
            print(50*'-', '\n', sep='')

            ## cleanup memory
            del Par
        
    ## otherwise
    else:

        ## generate parameters
        print('\n', 50*'-', sep='')
        print('precalibrating')
        Par = precalib_func(**kwargs)

        ## save output
        print('saving')
        Par.to_netcdf(path_precalib_out / f'{name}.nc', encoding={var:{'zlib':True, 'dtype':np.float32} for var in Par})
        print(50*'-', '\n', sep='')

        ## cleanup memory
        del Par

