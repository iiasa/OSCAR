"""
OSCAR - Customized Workflow (Level 2)
Action: Executes user-defined research projects through a gated 4-step pipeline.
"""

import yaml
import numpy as np
import xarray as xr
import pandas as pd
from pathlib import Path
# Local imports
from oscar._utils.load_config import load_config
from oscar._io.paths import get_paths
from oscar._io.handlers import compile_custom_forcing
from oscar._io._download import ensure_configured_library
from oscar._core._model.OSCAR import OSCAR
from oscar._utils.metadata import apply_variable_metadata

def _flatten(nested):
    flat = []
    for item in nested:
        if isinstance(item, list): flat.extend(_flatten(item))
        else: flat.append(item)
    return flat

def _validate(val, allowed, name):
    flat = _flatten(allowed)
    if val not in flat: raise ValueError(f"Invalid {name}: '{val}'. Allowed: {flat}")

def run_customized(project_name=None, experiment=None, **kwargs):
    # --- 0. INIT & CONFIG ---
    if not project_name : raise ValueError("Project name required.")
    p_path = get_paths()["projects"] / project_name
    exp_label = experiment or project_name
    with open(p_path / f"settings_{exp_label}.yaml", "r") as f: u_cfg = yaml.safe_load(f)
    
    cfg = load_config()
    reg, c_mode, t1 = cfg['registry'], cfg['customized_mode'], cfg['configured_mode']
    marker_s = c_mode.get('marker_scenarios')

    experiment_id = u_cfg.get('experiment_name', exp_label)
    s_sci = u_cfg.get('scientific_setup', {})
    hist_t = s_sci.get('hist_type', t1['defaults']['hist_type'])
    reg_run = s_sci.get('model_region', t1['defaults']['region'])
    base_s = s_sci.get('baseline_forcing', t1['defaults']['scenario'][0])
    end_yr = s_sci.get('projection_end_year', t1['run_end'])

    _validate(hist_t, c_mode['allowed_hist'], "hist_type")
    _validate(reg_run, c_mode['allowed_regions'], "model_region")
    _validate(base_s, c_mode['allowed_baselines'], "baseline_forcing")
    
    h_spec = reg['hist_versions'][hist_t]
    h_end, s_start, n_mc = h_spec['hist_end'], h_spec['scen_start'], t1['official_nMC']
    lib_path = ensure_configured_library(hist_t, reg_run)

    p_forcing_file = p_path / f"forcing_processed_{experiment_id}.nc"
    results_file = p_path / "results" / f"{experiment_id}_results.nc"
    audit_dir, reg_file = p_path / "forcing_audit", p_path / "forcing_audit" / "source_registry.csv"
    theme_vars = _resolve_vars(u_cfg.get('theme', 'climate'), u_cfg.get('custom_vars'), c_mode)

    # --- STEP 1: FORCING ---
    if u_cfg.get('preprocess_forcing', True):
        u_files = u_cfg.get('user_files', {})
        
        # 1. Load or Compile User Forcing
        ds_user = xr.open_dataset(p_path / u_files['compiled_nc']).load() if u_files.get('compiled_nc') else \
                  compile_custom_forcing(p_path, u_files.get('csv_inputs', {}), mod_region=reg_run)
        
        # 2. Create Source Registry for Audit Plotting
        sources = []
        audit_dir.mkdir(exist_ok=True, parents=True)

        for sn in ds_user.scen.values:
            u_sn = ds_user.sel(scen=sn, drop=True)
            for var in u_sn.data_vars:
                # Check if the variable has a species/halo dimension
                # Species dimension names commonly include 'spc', 'halo', or 'species'
                spc_dim = [d for d in u_sn[var].dims if 'spc' in d or 'halo' in d]
                
                if spc_dim:
                    # If it's a multi-species variable (like E_Xhalo), list each species
                    for s_val in u_sn[var][spc_dim[0]].values:
                        sources.append({
                            'scenario': str(sn),
                            'variable': str(var),
                            'species': str(s_val)
                        })
                else:
                    # Global variables (like E_CH4 or Eff) have no species
                    sources.append({
                        'scenario': str(sn),
                        'variable': str(var),
                        'species': 'None'
                    })
        # Save to CSV for the Audit Plotter to read
        pd.DataFrame(sources).to_csv(reg_file, index=False)
        #print(f"[OSCAR] Source registry created: {reg_file}")

        # 3. Load Library Forcing (History and Scenarios)
        with xr.open_dataset(lib_path/"forcing_hist.nc") as tmp: 
            ds_h = tmp.load()
        with xr.open_dataset(lib_path/"forcing_scen.nc") as tmp: 
            ds_s = tmp.load()

        # Define time bounds
        f_anchor = ds_h.sel(year=h_end, drop=True)
        f_years = np.arange(s_start, end_yr + 1)
        
        # Create a "Template" from the base scenario defined in settings
        ds_base = ds_s.sel(scen=base_s, drop=True).sel(year=f_years)

        # 4. Process each user scenario and merge with the baseline
        user_runs = []
        for sn in ds_user.scen.values:
            # Select the scenario; u_sn now has dimensions (year, ...) but no 'scen' coordinate
            u_sn = ds_user.sel(scen=sn, drop=True)
            
            # --- Connection Logic ---
            if h_end in u_sn.year.values:
                if s_sci.get('connect_method') == 'scaling':
                    u_val_at_anchor = u_sn.sel(year=h_end)
                    # Ensure f_anchor and u_val_at_anchor are aligned for the division
                    scale_factor = xr.where(u_val_at_anchor != 0, f_anchor / u_val_at_anchor, 1.0)
                    u_sn = u_sn * scale_factor
                u_sn = u_sn.sel(year=slice(s_start, None))

            # --- MERGE LOGIC ---
            # Start with a copy of the baseline (which is also single-scenario)
            merged = ds_base.copy(deep=True)
            
            # Remove 'scen' from the baseline copy if it exists, to match u_sn's structure
            if 'scen' in merged.coords:
                merged = merged.drop_vars('scen')

            # Loop through variables: use User data if provided, else keep Baseline
            for var in ds_base.data_vars:
                if var in u_sn.data_vars:
                    # combine_first uses u_sn values where available, 
                    # and fills with merged (baseline) values elsewhere
                    merged[var] = u_sn[var].combine_first(merged[var])
            
            # Final safety: fill any remaining NaNs using the original baseline
            merged = merged.fillna(ds_base.drop_vars('scen') if 'scen' in ds_base.coords else ds_base)
            
            # Join historical anchor (2014) + projection (2015+)
            anchor_entry = f_anchor.expand_dims(year=[h_end])
            
            # Reindex the anchor to match the species in the merged data
            anchor_entry = anchor_entry.reindex_like(merged.isel(year=0), method=None, fill_value=0)

            full = xr.concat([anchor_entry, merged], dim='year').interp(year=np.arange(h_end, end_yr + 1))
            
            # Re-attach the scenario label before adding to the list
            user_runs.append(full.sel(year=f_years).expand_dims(scen=[sn]))
        
        # 4. Create Reference Scenario (using marker scenarios)
        for m_sn in marker_s:
            # Select the marker scenario, rename it by adding '-ref', and append it to the existing user_runs list
            user_runs.append(ds_s.sel(scen=m_sn, drop=True).sel(year=f_years).expand_dims(scen=[f"{m_sn}-ref"]))
        
        # 4. Final Combination
        for_final = xr.concat(user_runs, dim='scen')

        # 5. Drop concentration-driven variables to avoid confusion (since user inputs are in emissions)
        # These vars are D_CO2, D_CH4, D_N2O, D_Xhalo
        # Note: it is hard-coded that this mode cannot be driven by concentration scenarios
        for_final = for_final.drop_vars([var for var in for_final if var in ['D_CO2', 'D_CH4', 'D_N2O', 'D_Xhalo']])
        #print(for_final.Eff.isel(scen=1)-for_final.Eff.isel(scen=12))
        if p_forcing_file.exists(): 
            p_forcing_file.unlink()
        # Save
        for_final.to_netcdf(p_forcing_file, engine="h5netcdf")

    # --- STEP 2: AUDIT ---
    if u_cfg.get('plot_user_forcing', True):
        # load the processed user forcing and the historical library forcing for comparison
        p_forcing_file = p_path / f"forcing_processed_{experiment_id}.nc"
        from oscar._viz import plot_forcing_audit
        plot_forcing_audit(xr.open_dataset(p_forcing_file), xr.open_dataset(lib_path/"forcing_hist.nc").load(), reg_file, audit_dir)

    # --- STEP 3: RUN ---
    if u_cfg.get('run_model', True):
        p, ini, h_res = [xr.open_dataset(lib_path/f"{x}_nMC{n_mc}.nc").load() for x in ['params', 'ini_state', 'hist_results']]
        with xr.open_dataset(p_forcing_file) as tmp:
            for_scen=tmp.load()
        out_s = OSCAR(Ini=ini, Par=p, For=for_scen, nt=4, var_keep=theme_vars, **kwargs)
        
        res = apply_variable_metadata(xr.concat([h_res[theme_vars], out_s[theme_vars]], dim='year'))
        results_file.parent.mkdir(exist_ok=True); res.to_netcdf(results_file, engine="h5netcdf", mode='w')

    # --- STEP 4: PLOT & DISPLAY ---
    ds_final = xr.open_dataset(results_file)
    if u_cfg.get('plot_outputs', True):
        from oscar._viz import plot_ts_scen
        print(f"[OSCAR] Step 4: Generating summary plots...")
        
        # 1. Global aggregation
        ds_p = ds_final.sum('reg_land', keep_attrs=True) if 'reg_land' in ds_final.dims else ds_final

        # IMPORTANT: Ensure the year index is unique before passing to the plotter
        ds_p = ds_p.drop_duplicates("year")

        # Skip the manual broadcast/concat entirely
        plot_ts_scen(
            ds=ds_p, 
            split_year=h_end, 
            var_list=theme_vars, 
            out_dir=results_file.parent, 
            show_plot=True
        )
    print("\n" + "="*60)
    print(f"CUSTOMIZED RUN COMPLETE\n  Project    : {project_name}\n  Experiment : {experiment_id}\n  Saved in   : {results_file.absolute()}")
    print("="*60 + "\n")
    #print(ds_final)
    return ds_final

def _resolve_vars(theme, custom, c_mode):
    if theme == "custom": return custom
    return _flatten(c_mode['output_themes'].get(theme, c_mode['output_themes']['climate']))