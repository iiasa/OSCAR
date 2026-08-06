"""
OSCAR Visualization: Forcing Audit Suite
Architecture: 100% Independent category plotters.
"""
import matplotlib.pyplot as plt
from matplotlib.units import registry
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
import yaml
from oscar._io.paths import PACKAGE_ROOT
from oscar._utils.load_config import load_config

def plot_forcing_audit(forcing_ds, forcing_hist, user_vars, out_dir, # user_vars is the path to source_registry.csv
                       plot_emissions=True, plot_halogens=True, plot_rf=False, plot_lulcc=False, plot_conc=False):
    """Main entry point: Calls independent category plotters."""
    if plot_emissions: _plot_emissions(forcing_ds, forcing_hist, user_vars, out_dir)
    if plot_lulcc:     _plot_lulcc(forcing_ds, forcing_hist, user_vars, out_dir)
    if plot_rf:        _plot_rf(forcing_ds, forcing_hist, user_vars, out_dir)
    if plot_halogens:  _plot_halogens(forcing_ds, forcing_hist, user_vars, out_dir) # Placeholder for future development
    if plot_conc:      _plot_conc(forcing_ds, forcing_hist, user_vars, out_dir) # Placeholder for future development

# --- HELPER: ADD MASTER LEGEND ---
def _add_audit_legend(fig):
    lines = [
        plt.Line2D([0], [0], color='tab:gray', lw=2, linestyle='-', label='Historical'),
        plt.Line2D([0], [0], color='tab:green', lw=2, linestyle='-', label='User Scenario'),
        plt.Line2D([0], [0], color='tab:gray', lw=2, linestyle='--', label='Library Fill'),
        plt.Line2D([0], [0], color='tab:blue', lw=2, linestyle=':', label='Marker (-ref)')
    ]
    fig.legend(handles=lines, loc='upper center', ncol=4, frameon=False, fontsize=10)

# --- Uni Handler
def _get_oscar_unit(var_name):
    """
    Retrieves the unit string from input_species.yaml for any OSCAR variable.
    """
    input_var_path = PACKAGE_ROOT / "oscar" / "_utils" /"_resources" / "input_species.yaml"
    
    try:
        with open(input_var_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
            
        # Search across all categories in the YAML
        for cat in ['anthropogenic_emissions', 'lulcc', 'rf', 'conc']:
            cat_dict = cfg.get(cat, {})
            if var_name in cat_dict:
                return cat_dict[var_name].get('unit', 'units')
                
    except Exception as e:
        print(f"[!] Warning: Could not read units for {var_name}: {e}")
        
    # Fallback defaults
    fallbacks = {'D_Eant_Xhalo': 'Gg yr-1', 'D_Xhalo': 'ppt', 'D_Eff': 'PgC yr-1'}
    return fallbacks.get(var_name, "units")

# --- 1. EMISSIONS PLOTTER ---
def _plot_emissions(ds_s, ds_h, user_vars, out_dir):
    """
    Plots standard emissions. 
    Uses user_vars (path to source_registry.csv) to identify which scenario/var 
    combinations were user-provided.
    """
    # load variable list from the .yaml registry
    from oscar._utils.load_config import load_variable_names
    input_var_path = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "input_species.yaml"
    var_ant_emissions = load_variable_names(input_var_path, section="anthropogenic_emissions")
    # drop D_Eant_Xhalo from the list, as they are handled separately in halogen plotting
    var_ant_emissions = [v for v in var_ant_emissions if v != 'D_Eant_Xhalo']
    present = [v for v in var_ant_emissions if v in ds_s.data_vars]
    if not present: return

    # 1. LOAD THE SOURCE REGISTRY
    registry_path = Path(user_vars)
    if registry_path.exists():
        df_reg = pd.read_csv(registry_path)
        if df_reg.empty:
            user_pairs = set()  # Empty registry = no user variables
        else:
            # Create a set of (scenario, variable) tuples for high-speed lookup
            user_pairs = set(zip(df_reg['scenario'].astype(str), df_reg['variable'].astype(str)))
    else:
        print(f"[!] Warning: {registry_path.name} not found. Defaulting all to Library Fill.")
        user_pairs = set()

    fig, axes = plt.subplots(3, 4, figsize=(22, 10), sharex=True)
    axes_flat = axes.flatten()

    for i, var in enumerate(present):
        ax = axes_flat[i]
        
        # Pre-calculate global totals for speed
        var_h = ds_h[var].sum('reg_land') if 'reg_land' in ds_h[var].dims else ds_h[var]
        var_s = ds_s[var].sum('reg_land') if 'reg_land' in ds_s[var].dims else ds_s[var]

        # Plot Historical (Solid Gray)
        ax.plot(ds_h.year.values, var_h.values, color='tab:gray', linestyle='-', lw=1.5)

        # Plot Scenarios
        for sn in var_s.scen.values:
            sn_str = str(sn)
            
            # --- THE THREE-WAY COLOR LOGIC ---
            if sn_str.endswith("-ref"):
                color, ls, alpha, lw = 'tab:blue', ':', 0.5, 1.5   # Markers
            elif (sn_str, var) in user_pairs:
                color, ls, alpha, lw = 'tab:green', '-', 1.0, 1.5  # TRUE User Data
            else:
                color, ls, alpha, lw = 'tab:gray', '--', 0.6, 1.5  # Library Fill

            ax.plot(var_s.year.values, var_s.sel(scen=sn).values, 
                    color=color, linestyle=ls, lw=lw, alpha=alpha)
        
        ax.set_title(var, fontweight='bold')
        ax.grid(True, alpha=0.15)
        # Labels: Y-axis (Units) on left column, X-axis (Year) on bottom row
        ax.set_ylabel(_get_oscar_unit(var), fontsize=9)
        if i >= len(present) - 3:
            ax.set_xlabel("Year", fontsize=9)

    # Cleanup and Save
    for j in range(len(present), 12): fig.delaxes(axes_flat[j])
    _add_audit_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_dir / f"plot_forcing_emissions.png", dpi=120)
    plt.close()
    #print(f"[OSCAR] Generated emissions audit plot.")

def _plot_halogens(ds_s, ds_h, user_vars_path, out_dir):
    """
    Plots halogen emissions by chemical family.
    Uses source_registry.csv to identify which scenario/species 
    combinations were user-provided.
    """
    # Updated chemical family grouping (Total: 41 species)
    families = {"CFCs": ["CFC-11", "CFC-12", "CFC-113", "CFC-114", "CFC-115", "CCl4", "CH3CCl3"], #7
                "HCFCs": ["HCFC-22", "HCFC-141b", "HCFC-142b"], #3
                "HFCs": ["HFC-23", "HFC-32", "HFC-125", "HFC-134a", "HFC-143a", "HFC-152a", 
                         "HFC-227ea", "HFC-236fa", "HFC-245fa", "HFC-365mfc", "HFC-43-10mee"], #11
                "Others": ["Halon-1202", "Halon-1211", "Halon-1301", "Halon-2402", "CH3Br", "CH3Cl",
                           "CH2Cl2", "CHCl3", "SF6", "NF3", "SO2F2", "CF4", "C2F6", "C3F8",
                           "c-C4F8", "C4F10", "n-C5F12", "n-C6F14", "C7F16", "C8F18"]} #20
    
    if 'D_Eant_Xhalo' not in ds_s or 'D_Eant_Xhalo' not in ds_h:
        return
    
    # 1. LOAD THE SOURCE REGISTRY
    registry_path = Path(user_vars_path)
    user_keys = set()
    if registry_path.exists():
        df_reg = pd.read_csv(registry_path)
        if not df_reg.empty:
            df_reg = df_reg.fillna('None')
            user_keys = set(zip(df_reg['scenario'].astype(str), df_reg['variable'].astype(str), df_reg['species'].astype(str)))
    else:
        print(f"[!] Warning: Registry {registry_path.name} not found. Defaulting to gray.")
    # Get the unit string for the emissions
    unit_str = _get_oscar_unit('D_Eant_Xhalo')
    # Aggregate regional data to global total for auditing (Gg yr-1)
    da_s = ds_s['D_Eant_Xhalo'].sum('reg_land') if 'reg_land' in ds_s['D_Eant_Xhalo'].dims else ds_s['D_Eant_Xhalo']
    da_h = ds_h['D_Eant_Xhalo'].sum('reg_land') if 'reg_land' in ds_h['D_Eant_Xhalo'].dims else ds_h['D_Eant_Xhalo']
    for fam_name, members in families.items():
        # --- CRITICAL CHECK: Only plot species that exist in BOTH History and Scenario data ---
        present = [m for m in members if m in da_s.spc_halo.values and m in da_h.spc_halo.values]
        if not present:
            continue
        
        # Determine grid size (up to 4x4)
        n = len(present)
        cols = 4 if n <12 else 5
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols*4, rows*3), squeeze=False, sharex=True)
        axes_flat = axes.flatten()
        
        for i, spc in enumerate(present):
            ax = axes_flat[i]
            
            # 2. PLOT HISTORICAL BASELINE (Solid Gray)
            ax.plot(ds_h.year.values, da_h.sel(spc_halo=spc).values, color='tab:gray', linestyle='-', lw=1.5)
            
            # 3. PLOT SCENARIOS (Differentiated by Source)
            for sn in da_s.scen.values:
                sn_str = str(sn)
                
                # Check if this specific species/scenario combo was in the user CSV
                is_user = (sn_str, 'D_Eant_Xhalo', str(spc)) in user_keys
                
                if sn_str.endswith("-ref"):
                    color, ls, alpha, lw = 'tab:blue', ':', 0.6, 1.2   # Reference Markers
                elif is_user:
                    color, ls, alpha, lw = 'tab:green', '-', 1.0, 1.8  # User Input
                else:
                    color, ls, alpha, lw = 'tab:gray', '--', 0.6, 1.2  # Library Background Fill

                ax.plot(da_s.year.values, da_s.sel(spc_halo=spc, scen=sn).values, color=color, linestyle=ls, lw=lw, alpha=alpha)
            
            ax.set_title(spc, fontweight='bold', fontsize=11)
            if i % cols == 0:
                ax.set_ylabel(unit_str, fontsize=9)
            if i >= n - cols:
                ax.set_xlabel("Year", fontsize=9)
            ax.grid(True, alpha=0.15)
        
        # Remove empty subplots
        for j in range(n, len(axes_flat)):
            fig.delaxes(axes_flat[j])
        
        # Add the unified legend across all halogen PNGs
        _add_audit_legend(fig)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        save_path = out_dir / f"plot_forcing_halo_{fam_name}.png"
        plt.savefig(save_path, dpi=100, bbox_inches='tight')
        plt.close()
        #print(f"[OSCAR] Generated halogen audit: {fam_name}")

# --- 3. RADIATIVE FORCING PLOTTER (Fixed version) ---
def _plot_rf(ds_s, ds_h, user_vars_path, out_dir):
    vars = ['RF_solar', 'RF_volc', 'RF_contr']
    present = [v for v in vars if v in ds_s.data_vars]
    if not present: return

    # LOAD REGISTRY to check for user variables
    registry_path = Path(user_vars_path)
    user_vars_set = set()
    if registry_path.exists():
        df_reg = pd.read_csv(registry_path)
        user_vars_set = set(df_reg['variable'].unique())

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True)
    for i, var in enumerate(present):
        ax = axes[i]
        if var in ds_h:
            ax.plot(ds_h.year.values, ds_h[var].values, color='tab:gray', ls='-')

        for sn in ds_s.scen.values:
            sn_str = str(sn)
            # LOGIC FIX: Check against the loaded set, not the path
            if sn_str.endswith("-ref"): 
                color, ls, alpha = 'tab:blue', ':', 0.5
            elif var in user_vars_set: 
                color, ls, alpha = 'tab:green', '-', 1.0
            else: 
                color, ls, alpha = 'tab:gray', '--', 0.6
            
            ax.plot(ds_s.year.values, ds_s[var].sel(scen=sn).values, color=color, linestyle=ls)
        ax.set_title(var)
        ax.grid(True, alpha=0.15)

    _add_audit_legend(fig)
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(out_dir / f"plot_forcing_rf.png", dpi=120)
    plt.close()

# --- 4. LULCC PLOTTER ---
def _plot_lulcc(ds, user_vars, out_dir):
    print("[OSCAR] LULCC plotting not implemented yet. Placeholder for future development.")

# --- 5. CONCENTRATIONS PLOTTER ---
def _plot_conc(ds, user_vars, out_dir, output_identifier):
    print("[OSCAR] Concentration plotting not implemented yet. Placeholder for future development.")

