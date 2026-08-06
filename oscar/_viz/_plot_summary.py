"""
OSCAR Visualization Module
Time-series plotting for global variables.
"""
import sys
import subprocess
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use('Agg')  # set before importing pyplot, or at the very top of the script, to avoid issues on plots being shown twice
import matplotlib.pyplot as plt
import warnings

def plot_timeseries_summary(ds, split_year, var_list, out_dir, show_plot=True):
    """
    Plots historical vs scenario time-series with professional scientific titles.
    Format: Long Name | Sci Name | Var Name [Unit]
    """
    ### silent warning for some data missing now for last year scenario ###
    warnings.filterwarnings(
        "ignore",
        message="All-NaN slice encountered",
        category=RuntimeWarning,
    )
    ### silent warning for missing data in the last year of the scenario ###
    for var in var_list:
        if var not in ds:
            continue
        
        fig = plt.figure(figsize=(9, 6))
        
        # 1. Select the variable DataArray and squeeze extra dims (like region)
        # Squeezing here ensures we have a clean (year, config, [scen]) object
        da = ds[var].squeeze()
        
        # 2. Split Timeline
        h = da.sel(year=slice(None, split_year))
        s = da.sel(year=slice(split_year + 1, None))

        # 3. Plot Historical (Black line + ribbon)
        # pick the "historical" scenario if it exists, otherwise fall back to the first scenario
        if 'scen' in h.dims:
            scen_vals = [str(v) for v in h.scen.values]
            if 'Historical' in scen_vals:
                h_plot = h.sel(scen='Historical').squeeze()
            else:
                h_plot = h.isel(scen=0).squeeze()
        else:
            h_plot = h

        if 'config' in h_plot.dims:
            # Calculate median and 33/66 percentiles
            mh = h_plot.median('config')
            h_low = h_plot.quantile(0.10, dim='config')
            h_high = h_plot.quantile(0.90, dim='config')

            plt.plot(h_plot.year, mh, color='k', lw=2, label='Historical')
            plt.fill_between(h_plot.year, h_low, h_high, color='k', alpha=0.2)
        else:
            plt.plot(h_plot.year, h_plot, color='k', lw=2, label='Historical')

        # 4. Plot Scenario (Colored lines per scenario)
        scens = s.scen.values if 'scen' in s.dims else [None]
        # exclude the historical scenario if it exists
        scens = [sn for sn in scens if sn != 'Historical']
        for sn in scens:
            s_sub = s.sel(scen=sn) if sn is not None else s
            label = str(sn) if sn is not None else "Projection"
            
            if 'config' in s_sub.dims:
                # Calculate median and 33/66 percentiles
                ms = s_sub.median('config')
                s_low = s_sub.quantile(0.10, dim='config')
                s_high = s_sub.quantile(0.90, dim='config')
                
                line, = plt.plot(s.year, ms, lw=1.5, label=label)
                plt.fill_between(s.year, s_low, s_high, 
                                 color=line.get_color(), alpha=0.2)
            else:
                plt.plot(s.year, s_sub, lw=1.5, label=label)

        # --- 5. Professional Scientific Title Logic ---
        long_name = da.attrs.get('long_name', var)
        sci_name  = da.attrs.get('sci_name', '')
        units     = da.attrs.get('units', 'n/a')
        
        # Build a descriptive title: "Long Name (Symbol: ΔTg | ID: D_Tg)"
        title_str = f"{long_name}"
        if sci_name:
            title_str += f" ({sci_name} | ID: {var})"
        else:
            title_str += f" (ID: {var})"
        
        plt.title(title_str, fontsize=12, fontweight='normal', pad=10)
        plt.ylabel(f"[{units}]", fontsize=11)
        plt.xlabel("Year", fontsize=11)
        plt.legend(loc='upper left', fontsize='small', ncol=2 if len(scens) > 1 else 1)
        plt.grid(True, alpha=0.3)

        # 6. Save and Close
        plot_file = out_dir / f"plot_{var}.png"
        if show_plot:
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')
            if sys.platform == 'darwin':
                subprocess.run(['open', str(plot_file)])
        else:
            plt.savefig(plot_file, dpi=150, bbox_inches='tight')

        plt.close(fig)
