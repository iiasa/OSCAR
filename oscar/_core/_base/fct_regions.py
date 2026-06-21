import csv
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

from matplotlib.colors import hsv_to_rgb, ListedColormap

from oscar._io.paths import REGIONS_INFO_PATH
path_reg = REGIONS_INFO_PATH


##################################################
##   1. ANCILLARY FUNCTIONS
##################################################

## display chosen regional aggregation
def plot_regions(mod_region, old_axis='reg_code', new_axis='reg_land', rez='1deg'):

    ## read region mapping file
    reg_dict = pd.read_csv(path_reg / 'OSCAR_reg_dict.csv', index_col=old_axis)[[mod_region]]
    reg_dict = reg_dict.loc[:, mod_region].to_dict()

    ## read long name file
    with open(path_reg / 'OSCAR_reg_names.csv', newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if line and line[0] == mod_region:
                reg_names = {col.split(':')[0]: col.split(':')[1] for col in line[1:]}
                reg_order = [col.split(':')[0] for col in line[1:]]
                break

    ## read reg_code mask
    with xr.open_dataset(path_reg / f'OSCAR_region_mask_{rez}.nc') as TMP: ds_mask = TMP.load()
    mask = ds_mask.frac_reg.sum(old_axis, min_count=1) > 0
    
    ## make new regions
    ds_mask.coords[new_axis] = xr.DataArray([reg_dict[reg] for reg in ds_mask[old_axis].values], dims=old_axis)
    ds_mask = ds_mask.groupby(new_axis).sum(old_axis, min_count=1, keep_attrs=True)
    ds_mask = ds_mask.sel({new_axis: reg_order})
    ds_mask.coords[new_axis + '_name'] = xr.DataArray([reg_names[reg] for reg in ds_mask[new_axis].values], dims=new_axis)

    ## get main region in each gridcell
    ds = ds_mask.frac_reg.argmax(new_axis)
    ds = ds.where(mask)

    ## make distinct colormap
    def max_separated_colors(n, s=0.7, v=0.9):
        hues = np.linspace(0, 1, n, endpoint=False)
        stride = 0.61803398875
        indices = np.array([(i * stride) % 1 for i in range(n)])
        indices = np.argsort(indices)
        hsv_colors = np.stack([hues[indices], np.full(n, s), np.full(n, v)], axis=1)
        rgb_colors = hsv_to_rgb(hsv_colors)
        return ListedColormap(rgb_colors)

    ## make plot
    plt.figure(figsize=(8, 6))
    ax = plt.subplot(1, 1, 1)
    p = ds.plot(cmap=max_separated_colors(len(ds_mask[new_axis])), add_colorbar=True)
    
    ## edit colorbar
    cbar = p.colorbar
    cbar.set_ticks(np.arange(0, len(ds_mask[new_axis])))
    cbar.set_ticklabels([ds_mask[new_axis].values[n] + ': ' + ds_mask[new_axis + '_name'].values[n] for n in np.arange(0, len(ds_mask[new_axis]))])
    cbar.ax.tick_params(labelsize='x-small')
    cbar.set_label('')

    ## edit rest
    plt.xticks(fontsize='small')
    plt.yticks(fontsize='small')
    ax.set_xlabel(ax.get_xlabel(), fontsize='small')
    ax.set_ylabel(ax.get_ylabel(), fontsize='small')
    plt.title(f'OSCAR mod_region = {mod_region}', fontsize='small')
    plt.tight_layout()


##################################################
##   2. REGIONAL AGGREGATION
##################################################

## aggregate regional data to OSCAR regions
def aggreg_regions(ds_in, mod_region, weight_var={}, old_axis='reg_code', new_axis='reg_land', time_axis='year'):
    '''
    Function to aggregate data onto OSCAR regions. It uses dictionnaries mapping ISO regions to OSCAR regions defined in 'input_data/regions' by user.
    
    Input:
    ------
    ds_in (xr.Dataset)  input dataset to be aggregated
    mod_region (str)    name of regional aggregation (must be a valid option)
        
    Output:
    -------
    ds_out (xr.Dataset) output dataset

    Options:
    --------
    weight_var (dict)   keys variables are weighted using values variables when aggregating; 
                        this is necessary for intensive variables (e.g. temperature that needs to be weighted by area); 
                        keys and values are names (str) of ds_in variables;
                        default = {}
    old_axis (str)      name of regional axis that will be aggregated (must a dim of ds_in);
                        default = 'reg_code'
    new_axis (str)      name of new aggregated regional axis (must NOT be in ds_in, and will be in ds_out);
                        default = 'reg_land'
    time_axis (str)     name of time axis (to ensure it is first dim in ds_out);
                        default = 'year'
    '''
    
    ## check old axis in ds_in and new_axis not in ds_in
    assert old_axis in ds_in.coords and new_axis not in ds_in.coords
    
    ## make deep copy to be safe
    ds_out = ds_in.copy(deep=True)

    ## read region mapping file
    reg_dict = pd.read_csv(path_reg / 'OSCAR_reg_dict.csv', index_col=old_axis)[[mod_region]]
    reg_dict = reg_dict.loc[:, mod_region].to_dict()

    ## read long name file
    with open(path_reg / 'OSCAR_reg_names.csv', newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line in reader:
            if line and line[0] == mod_region:
                reg_names = {col.split(':')[0]: col.split(':')[1] for col in line[1:]}
                reg_order = [col.split(':')[0] for col in line[1:]]
                break

    ## apply weights to weighted variables
    for var in weight_var:
        ds_out[var] = ds_out[var] * ds_out[weight_var[var]]

    ## separate variables without regional axis
    ds_non = ds_out.drop_vars([var for var in ds_out if old_axis in ds_out[var].dims])
    ds_non = ds_non.drop_dims(old_axis)
    ds_out = ds_out.drop_vars([var for var in ds_out if old_axis not in ds_out[var].dims])

    ## new regional aggregation
    ds_out.coords[new_axis] = xr.DataArray([reg_dict[reg] for reg in ds_out[old_axis].values], dims=old_axis)
    ds_out = ds_out.groupby(new_axis).sum(old_axis, min_count=1, keep_attrs=True)

    ## clean up region axes
    ds_out = ds_out.combine_first(xr.DataArray([np.nan for _ in range(len(reg_order))], coords={new_axis: reg_order}, dims=[new_axis]))
    ds_out = ds_out.sel({new_axis: reg_order})

    ## add information
    ds_out.coords[new_axis + '_name'] = xr.DataArray([reg_names[reg] for reg in ds_out[new_axis].values], dims=new_axis)
    ds_out.coords[new_axis].attrs['mod_region'] = ds_out.coords[new_axis + '_name'].attrs['mod_region'] = mod_region

    ## remove weights (and add back attributes)
    for var in weight_var:
        ds_out[var] = ds_out[var] / ds_out[weight_var[var]]
        ds_out[var].attrs = ds_in[var].attrs

    ## merge with separated variables
    ds_out = xr.merge([ds_out, ds_non], join='outer', compat='no_conflicts')

    ## make sure time axis is first
    if time_axis in ds_out.dims: 
        ds_out = ds_out.transpose(*([time_axis] + [dim for dim in ds_out.dims if dim != time_axis]))
    
    ## return
    return ds_out

