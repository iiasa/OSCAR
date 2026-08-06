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
from oscar._io.paths import get_paths
from oscar._io.handlers import compile_custom_forcing
from oscar._io._download import ensure_configured_library
from oscar._utils.load_config import load_config
from oscar._utils.metadata import apply_variable_metadata

from oscar._core._model.OSCAR import OSCAR


def _flatten(nested):
    flat = []
    for item in nested:
        if isinstance(item, list): flat.extend(_flatten(item))
        else: flat.append(item)
    return flat

def _validate(val, allowed, name):
    flat = _flatten(allowed)
    if val not in flat: raise ValueError(f"Invalid {name}: '{val}'. Allowed: {flat}")

def run_customized(project=None, experiment=None, **kwargs):
    # --- 0. INIT & CONFIG ---
    if not project : raise ValueError("Project name required.")
    p_path = get_paths()["projects"] / project
    with open(p_path / f"settings_{experiment}.yaml", "r") as f: u_cfg = yaml.safe_load(f)
    
    cfg = load_config()
    reg, c_mode, t1 = cfg['registry'], cfg['customized_mode'], cfg['configured_mode']
    marker_s = c_mode.get('marker_scenarios')

    output_id = u_cfg.get('output_identifier', f"{project}-{experiment}")
    s_sci = u_cfg.get('scientific_setup', {})
    hist_t = s_sci.get('hist_type', t1['defaults']['hist_type'])
    reg_run = s_sci.get('model_region', t1['defaults']['region'])
    base_s = s_sci.get('baseline_forcing', t1['defaults']['scenario'][0])
    end_yr = s_sci.get('projection_end_year', t1['run_end'])

    allowed_baseline = cfg['registry']['all_scens']['scen6'] + cfg['registry']['all_scens']['scen7']

    _validate(hist_t, c_mode['allowed_hist'], "hist_type")
    _validate(reg_run, c_mode['allowed_regions'], "model_region")
    _validate(base_s, allowed_baseline, "baseline_forcing")
    
    h_spec = reg['hist_versions'][hist_t]
    h_end, s_start, n_mc = h_spec['hist_end'], h_spec['scen_start'], t1['official_nMC']
    lib_path = ensure_configured_library(hist_t, reg_run)

    p_forcing_file = p_path / "forcing" / f"forcing_processed_{output_id}.nc"
    results_file = p_path / "results" / f"{output_id}_results.nc"
    audit_dir, reg_file = p_path / "forcing" / f"audit_plots_{output_id}", p_path / "forcing" / f"audit_plots_{output_id}" / f"source_registry_{output_id}.csv"
    theme_vars = _resolve_vars(u_cfg.get('theme', 'climate'), u_cfg.get('custom_vars'), c_mode)

    # --- STEP 1: FORCING ---
    if u_cfg.get('preprocess_forcing', True):
        u_files = u_cfg.get('user_files', {})

        # 1.1 Load or Compile User Forcing
        ds_user = None

        if u_files.get('compiled_nc'):
            ds_user = xr.open_dataset(p_path / u_files['compiled_nc']).load()
        elif u_files.get('csv_inputs'):
            ds_user = compile_custom_forcing(
                p_path, u_files['csv_inputs'], mod_region=reg_run
            )
            if ds_user is not None:
                ds_user = ds_user.load()

        # Validate: treat empty Dataset as None
        if ds_user is None or len(ds_user.data_vars) == 0:
            ds_user = None
            print('[OSCAR] User forcing: none provided')
        else:
            print('[OSCAR] User forcing: loaded')
            # Ensure 'scen' coordinate exists if scenario dimension exists
            if (
                'scen' in ds_user.dims
                and 'scen' not in ds_user.coords
                and len(ds_user.dims['scen']) > 0
            ):
                ds_user = ds_user.assign_coords(scen=ds_user['scen'].values)
            elif 'scen' not in ds_user.dims and 'scen' not in ds_user.coords:
                ds_user = ds_user.expand_dims(scen=['user_scen'])

        # 1.2 Create Source Registry for Audit Plotting
        sources = []
        audit_dir.mkdir(exist_ok=True, parents=True)

        if ds_user is not None and (
            'scen' in ds_user.coords or 'scen' in ds_user.dims
        ):
            for sn in ds_user.scen.values:
                u_sn = ds_user.sel(scen=sn, drop=True)
                for var in u_sn.data_vars:
                    spc_dim = [
                        d for d in u_sn[var].dims if 'spc' in d or 'halo' in d
                    ]

                    if spc_dim:
                        for s_val in u_sn[var][spc_dim[0]].values:
                            sources.append({
                                'scenario': str(sn),
                                'variable': str(var),
                                'species': str(s_val),
                            })
                    else:
                        sources.append({
                            'scenario': str(sn),
                            'variable': str(var),
                            'species': 'None',
                        })

        # Always write with headers
        pd.DataFrame(sources, columns=['scenario', 'variable', 'species']).to_csv(
            reg_file, index=False
        )

        # 1.3 Load Library Forcing (History and Scenarios)
        with xr.open_dataset(lib_path / 'forcing_hist.nc') as tmp:
            ds_h = tmp.load()
        with xr.open_dataset(lib_path / 'forcing_scen.nc') as tmp:
            ds_s = tmp.load()

        # Define time bounds
        f_anchor = ds_h.sel(year=h_end, drop=True)
        f_years = np.arange(s_start, end_yr + 1)

        # Helper function: safely extract reference variable slice handling scenarios
        def get_ref_da(ds_ref, var_name, scen_name, target_years):
            if var_name not in ds_ref.data_vars:
                return None
            da = ds_ref[var_name]
            # Check if reference variable has 'scen' dimension
            if 'scen' in da.dims:
                if scen_name in da.scen.values:
                    da = da.sel(scen=scen_name, drop=True)
                elif base_s in da.scen.values:
                    da = da.sel(scen=base_s, drop=True)
                else:
                    da = da.isel(scen=0, drop=True)
            if 'year' in da.dims:
                da = da.sel(year=target_years)
            return da

        # 1.4 Process each user scenario
        user_runs = []

        if ds_user is not None and (
            'scen' in ds_user.coords or 'scen' in ds_user.dims
        ):
            for sn in ds_user.scen.values:
                u_sn = ds_user.sel(scen=sn, drop=True)

                # Connection Logic (Variable-by-Variable Scaling)
                if h_end in u_sn.year.values:
                    if s_sci.get('connect_method') == 'scaling':
                        for v in u_sn.data_vars:
                            if v in f_anchor.data_vars:
                                u_anc = u_sn[v].sel(year=h_end)
                                f_anc = f_anchor[v]
                                scale = xr.where(
                                    (u_anc != 0) & (~np.isnan(u_anc)),
                                    f_anc / u_anc,
                                    1.0,
                                )
                                u_sn[v] = u_sn[v] * scale

                    u_sn = u_sn.sel(year=slice(s_start, None))

                # --- CREATE EMPTY TEMPLATE AND FILL WITH USER INPUT / REFERENCE ---
                merged = xr.Dataset()
                all_vars = set(ds_s.data_vars) | set(u_sn.data_vars)

                for var in all_vars:
                    ref_da = get_ref_da(ds_s, var, base_s, f_years)
                    u_var = u_sn[var] if var in u_sn.data_vars else None
                    if var == 'D_Eant_Xhalo':
                        u_var = u_var.sum('reg_land') if u_var is not None else None

                    # Step A: Create Empty NaN Template based on Reference or User grid
                    if ref_da is not None:
                        template = xr.full_like(ref_da, fill_value=np.nan)
                    elif u_var is not None:
                        u_var_sliced = (
                            u_var.sel(year=f_years)
                            if 'year' in u_var.dims
                            else u_var
                        )
                        template = xr.full_like(u_var_sliced, fill_value=np.nan)
                    else:
                        continue

                    # Step B: Fill with User Input if provided
                    if u_var is not None:
                        # Align coordinate types (e.g. reg_land int vs str or int32 vs int64)
                        for dim in template.dims:
                            if (
                                dim in u_var.dims
                                and u_var[dim].dtype != template[dim].dtype
                            ):
                                if u_var[dim].size == template[dim].size:
                                    u_var = u_var.assign_coords(
                                        {dim: template[dim].values}
                                    )

                        u_var_aligned = u_var.reindex_like(template)
                        template = xr.where(
                            ~np.isnan(u_var_aligned), u_var_aligned, template
                        )

                    # Step C: Fill remaining missing values (NaNs) from Reference
                    if ref_da is not None:
                        ref_da_aligned = ref_da.reindex_like(template)
                        template = xr.where(
                            np.isnan(template), ref_da_aligned, template
                        )

                    merged[var] = template

                # Re-anchor historical transition point at h_end
                anchor_entry = f_anchor.expand_dims(year=[h_end])
                anchor_entry = anchor_entry.reindex_like(
                    merged.isel(year=0), method=None, fill_value=0
                )

                full = xr.concat([anchor_entry, merged], dim='year').interp(
                    year=np.arange(h_end, end_yr + 1)
                )
                user_runs.append(full.sel(year=f_years).expand_dims(scen=[sn]))

        # 1.5 Create Reference Scenarios (using marker scenarios)
        for m_sn in marker_s:
            user_runs.append(
                ds_s.sel(scen=m_sn, drop=True)
                .sel(year=f_years)
                .expand_dims(scen=[f'{m_sn}-ref'])
            )

        # 1.6 Final Combination
        for_final = xr.concat(user_runs, dim='scen')

        # 1.7 Drop concentration-driven variables to avoid confusion
        for_final = for_final.drop_vars(
            [
                var
                for var in for_final
                if var in ['D_CO2', 'D_CH4', 'D_N2O', 'D_Xhalo']
            ]
        )

        if p_forcing_file.exists():
            p_forcing_file.unlink()

        # Save
        for_final.to_netcdf(p_forcing_file, engine='h5netcdf')
        print(f'[OSCAR] Step 1: User forcing processed and saved in {p_forcing_file}.')
    else:
        print('[OSCAR] Step 1: Forcing preprocessing skipped.')

    # --- STEP 2: AUDIT ---
    if u_cfg.get('plot_user_forcing', True):
        # load the processed user forcing and the historical library forcing for comparison
        from oscar._viz import plot_forcing_audit
        plot_forcing_audit(xr.open_dataset(p_forcing_file), xr.open_dataset(lib_path/"forcing_hist.nc").load(), reg_file, audit_dir)
        print(f"[OSCAR] Step 2: Forcing audit plots saved in {audit_dir}")
    else:
        print(f"[OSCAR] Step 2: Forcing audit skipped. No audit plots generated.")

    # --- STEP 3: RUN ---
    if u_cfg.get('run_model', True):
        p, ini, h_res = [xr.open_dataset(lib_path/f"{x}_nMC{n_mc}.nc").load() for x in ['params', 'ini_state', 'hist_results']]
        with xr.open_dataset(p_forcing_file) as tmp:
            for_scen=tmp.load()
        out_s = OSCAR(Ini=ini, Par=p, For=for_scen, var_keep=theme_vars, **kwargs)

        h_res = h_res.drop_vars('scen').expand_dims(scen=['Historical'])
        res = apply_variable_metadata(xr.concat([h_res[theme_vars], out_s[theme_vars]], dim='year', join='outer'))
        results_file.parent.mkdir(exist_ok=True); res.to_netcdf(results_file, engine="h5netcdf", mode='w')
        print(f"[OSCAR] Step 3: Model run complete. ")
    else:
        print(f"[OSCAR] Step 3: Model run skipped. Results not generated.")

    # --- STEP 4: PLOT & DISPLAY ---
    ds_final = xr.open_dataset(results_file)
    if u_cfg.get('plot_outputs', True):
        from oscar._viz import plot_timeseries_summary
        print(f"[OSCAR] Step 4: Generating summary plots...")
        
        # 1. Global aggregation
        ds_p = ds_final.sum('reg_land', keep_attrs=True) if 'reg_land' in ds_final.dims else ds_final

        # IMPORTANT: Ensure the year index is unique before passing to the plotter
        ds_p = ds_p.drop_duplicates("year")

        # Skip the manual broadcast/concat entirely
        plot_dir = results_file.parent / f"summary_plots_{output_id}"
        plot_dir.mkdir(exist_ok=True, parents=True)
        plot_timeseries_summary(ds=ds_p, split_year=h_end, var_list=theme_vars, out_dir=plot_dir, show_plot=True)
        print(f"[OSCAR] Step 4: Summary plots generated.")
    else:
        print(f"[OSCAR] Step 4: Plotting skipped. No summary plots generated.")

    # Print final summary
    print("\n" + "="*60)
    print(f"CUSTOMIZED RUN COMPLETE\n  Project    : {project}\n  Experiment : {experiment}-{output_id}")
    print(f"  Results saved in : {results_file.absolute()}")
    print("="*60 + "\n")
    #print(ds_final)
    return print(f"[OSCAR] Results dataset:\n {ds_final}")

def _resolve_vars(theme, custom, c_mode):
    if theme == "custom": return custom
    return _flatten(c_mode['output_themes'].get(theme, c_mode['output_themes']['climate']))