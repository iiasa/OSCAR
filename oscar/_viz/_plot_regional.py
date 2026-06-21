def plot_regional_comparison(ds_all, var_list, out_dir, split_year):
    """
    Creates a multi-panel plot for each variable.
    Panel 1: Global Total | Panels 2-N: Each Region.
    Plots User Scenarios vs. Reference Scenarios.
    """
    for var in var_list:
        regions = ds_all.reg_land.values
        n_reg = len(regions)
        
        # Grid: 1 (Global) + N (Regions). Determine layout.
        n_plots = n_reg + 1
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4), sharex=True)
        axes_flat = axes.flatten()

        # --- Subplot 0: Global Total ---
        ax_glob = axes_flat[0]
        _draw_comparison(ds_all.sum('reg_land'), var, ax_glob, split_year, "Global Total")

        # --- Subplots 1-N: Individual Regions ---
        for i, reg_id in enumerate(regions):
            ax = axes_flat[i+1]
            _draw_comparison(ds_all.sel(reg_land=reg_id), var, ax, split_year, f"Region: {reg_id}")

        # Cleanup empty subplots
        for j in range(n_plots, len(axes_flat)): fig.delaxes(axes_flat[j])
        
        plt.tight_layout()
        plt.savefig(out_dir / f"comparison_{var}.png")
        plt.close()

def _draw_comparison(da_sub, var, ax, split_year, title):
    """Internal helper to plot black hist + colored scenarios."""
    # Split Hist/Scen
    h = da_sub[var].sel(year=slice(None, split_year))
    s = da_sub[var].sel(year=slice(split_year + 1, None))
    
    # Plot Hist
    ax.plot(h.year, h.mean('config'), color='black', label='Historical')
    
    # Plot Scenarios (User + Reference)
    for sn in s.scen.values:
        mean = s.sel(scen=sn).mean('config')
        # Distinguish Reference Scens (dashed) from User Scens (solid)
        style = '--' if "SSP" in str(sn) else '-' 
        ax.plot(s.year, mean, label=str(sn), linestyle=style)
        
    ax.set_title(title)
    ax.grid(True, alpha=0.2)