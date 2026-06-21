"""
OSCAR Visualization Module
Time-series plotting for global variables.
"""
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

def plot_timeseries_summary(ds, split_year, var_list, out_dir, show_plot=True):
    """
    Plots historical vs scenario time-series with professional scientific titles.
    Format: Long Name | Sci Name | Var Name [Unit]
    """
    for var in var_list:
        if var not in ds:
            continue
        
        plt.figure(figsize=(9, 6))
        
        # 1. Select the variable DataArray and squeeze extra dims (like region)
        # Squeezing here ensures we have a clean (year, config, [scen]) object
        da = ds[var].squeeze()
        
        # 2. Split Timeline
        h = da.sel(year=slice(None, split_year))
        s = da.sel(year=slice(split_year + 1, None))

        # 3. Plot Historical (Black line + ribbon)
        # pick the first scenario if 'scen' exists to avoid multiple black lines
        h_plot = h.isel(scen=0) if 'scen' in h.dims else h
        
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
        plot_file = out_dir / f"plt_{var}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
        
        if show_plot:
            plt.show()
            plt.close()
        else:
            plt.close()

def plot_ts_scen(ds, split_year, var_list, out_dir, show_plot=True):
    """
    Plots historical vs scenario time-series.
    Calculates Median and Min-Max range across the 'config' dimension.
    Each 'scen' is plotted as a separate color.
    """
    for var in var_list:
        if var not in ds:
            continue
        
        plt.figure(figsize=(10, 6))
        
        # 1. Select the variable and clean the time index
        da = ds[var].squeeze()
        if "year" in da.indexes and not da.indexes["year"].is_unique:
            da = da.drop_duplicates("year")
        
        # 2. Split Timeline
        h = da.sel(year=slice(None, split_year))
        s = da.sel(year=slice(split_year + 1, None))

        # 3. Plot Historical (Black Median + Gray Range)
        # Usually historical doesn't have scenarios, so we take the first if it does
        h_plot = h.isel(scen=0) if 'scen' in h.dims else h
        
        if 'config' in h_plot.dims:
            h_med = h_plot.median('config')
            h_min = h_plot.min('config')
            h_max = h_plot.max('config')
            
            plt.plot(h_plot.year, h_med, color='k', lw=2, label='Historical', zorder=5)
            plt.fill_between(h_plot.year, h_min, h_max, color='k', alpha=0.15, zorder=4)
        else:
            plt.plot(h_plot.year, h_plot, color='k', lw=2, label='Historical', zorder=5)

        # 4. Plot Scenarios (Colored Median + Shaded Range per scenario)
        if 'scen' in s.dims:
            scens = s.scen.values
            for sn in scens:
                s_sub = s.sel(scen=sn)
                label = str(sn)
                
                if 'config' in s_sub.dims:
                    s_med = s_sub.median('config')
                    s_min = s_sub.min('config')
                    s_max = s_sub.max('config')
                    
                    line, = plt.plot(s_sub.year, s_med, lw=1.5, label=label)
                    plt.fill_between(s_sub.year, s_min, s_max, 
                                     color=line.get_color(), alpha=0.2)
                else:
                    plt.plot(s_sub.year, s_sub, lw=1.5, label=label)
        else:
            # Fallback if no scenario dimension exists
            if 'config' in s.dims:
                s_med = s.median('config')
                s_min = s.min('config')
                s_max = s.max('config')
                plt.plot(s.year, s_med, lw=1.5, label="Projection", color='blue')
                plt.fill_between(s.year, s_min, s_max, color='blue', alpha=0.2)
            else:
                plt.plot(s.year, s, lw=1.5, label="Projection", color='blue')

        # 5. Scientific Title and Labels
        long_name = da.attrs.get('long_name', var)
        sci_name  = da.attrs.get('sci_name', '')
        units     = da.attrs.get('units', 'n/a')
        
        title_str = f"{long_name}"
        if sci_name:
            title_str += f" ({sci_name} | ID: {var})"
        else:
            title_str += f" (ID: {var})"
        
        plt.title(title_str, fontsize=12, fontweight='normal', pad=10)
        plt.ylabel(f"[{units}]", fontsize=11)
        plt.xlabel("Year", fontsize=11)
        
        # Legend: num columns depends on how many scenarios we have
        num_scens = len(scens) if 'scen' in s.dims else 1
        plt.legend(loc='upper left', fontsize='small', ncol=2 if num_scens > 1 else 1)
        plt.grid(True, alpha=0.25)

        # 6. Save and Close
        plot_file = out_dir / f"ts_scen_{var}.png"
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"Plot saved: {plot_file}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()