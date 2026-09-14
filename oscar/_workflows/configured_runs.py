"""
OSCAR - Configured Workflow
Scientific projections using official pre-compiled CMIP libraries.
"""
import xarray as xr

from oscar._io.paths import get_paths
from oscar._io._download import ensure_configured_library
from oscar._utils.load_config import load_config
from oscar._utils.metadata import apply_variable_metadata

from oscar._core._model.OSCAR import OSCAR

def run_configured(
    scenario=None,  # user-selected scenario(s)
    region=None,    # user-selected region
    hist_type=None, # user-selected historical dataset
    variables=None, # user-selected output variables
    show_plot=True, # summary plot is shown by default
    run_model=True, # model is run by default
    # other kwargs are passed to OSCAR
    **kwargs
):
    # 1. LOAD CONFIGURATION (Source of Truth)
    cfg_full = load_config()
    registry = cfg_full['registry']
    cfg_mode = cfg_full['configured_mode']
    defaults = cfg_mode['defaults']
    
    # 2. RESOLVE FINAL SELECTIONS
    # Pull from user input OR Configured Mode defaults
    hist_final   = hist_type or defaults['hist_type']
    region_final = region    or defaults['region']
    
    scen_input   = scenario or defaults['scenario']
    scen_final   = [scen_input] if isinstance(scen_input, str) else list(scen_input)
    
    # Map 'var_select' from defaults to match function signature 'variables'
    vars_input   = variables or defaults.get('var_select', defaults.get('variables'))
    vars_final   = [vars_input] if isinstance(vars_input, str) else list(vars_input)

    # Resolve Scientific Years from Registry
    hist_specs      = registry['hist_versions'][hist_final]
    hist_end_year   = hist_specs['hist_end']
    scen_start_year = hist_specs['scen_start']
    scen_end_year   = cfg_mode['run_end'] # Official build ceiling (2100)
    
    n_mc = cfg_mode['official_nMC']

    # define allowed scenarios
    scen_nr = hist_final[-1]  # Extracts the number from 'CMIP6' or 'CMIP7'
    allowed_scenarios = registry['all_scens'][f'scen{scen_nr}']


    # Setup Output Path
    out_dir = get_paths()['results'] / "configured_run"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"oscar_configured_{hist_final}_{region_final}.nc"

    if run_model:
        # 3. VALIDATION (Check against allowed lists in configured_mode)
        _validate_choice(hist_final, cfg_mode['allowed_hist'], "hist_type")
        _validate_choice(region_final, cfg_mode['allowed_regions'], "region")
        for s in scen_final:
            _validate_choice(s, allowed_scenarios, "scenario")

        # 4. LOAD LIBRARY COMPONENTS
        print(f"Loading official library for {region_final} ({n_mc} members)...")
        # ensure_configured_library handles pathing and Zenodo downloads
        lib_path = ensure_configured_library(hist_final, region_final)
        
        # Load pre-built components (Using NetCDF4/HDF5)
        hist_results = xr.open_dataset(lib_path / f"hist_results_nMC{n_mc}.nc").load()
        ini_state    = xr.open_dataset(lib_path / f"ini_state_nMC{n_mc}.nc").load()
        scen_forcing = xr.open_dataset(lib_path / "forcing_scen.nc").load()
        params       = xr.open_dataset(lib_path / f"params_nMC{n_mc}.nc").load()

        print(f"Library components loaded from: {lib_path}")
        
        # 5. PREPARE INPUTS
        For_scen = scen_forcing.sel(scen=scen_final,year=slice(scen_start_year, scen_end_year))
        
        # 6. EXECUTE PROJECTION
        print(f"Running OSCAR (configured mode) for scenarios: {scen_final}")
        # nt=4 for parallel processing
        Out_scen = OSCAR(Ini=ini_state, Par=params, For=For_scen, var_keep=vars_final, **kwargs)
        
        # 7. COMBINE & APPLY METADATA
        print("Concatenating with history and applying metadata...")
        # Slice historical results to only include user-requested variables
        hist_results = hist_results.drop_vars('scen').expand_dims(scen=['Historical'])
        Out_all = xr.concat([hist_results[vars_final], Out_scen[vars_final]], dim='year')
        Out_all = apply_variable_metadata(Out_all)

        # 8. SAVE
        Out_all.to_netcdf(out_file, engine="h5netcdf")
        print(f"Success! Data saved to: {out_file}")
    
    else:
        # --- PLOT ONLY MODE ---
        if not out_file.exists():
            raise FileNotFoundError(f"No results found for {hist_final}/{region_final}. Set run_model=True.")
        Out_all = xr.open_dataset(out_file).load()

    # 9. PLOT SUMMARY
    from oscar._viz import plot_timeseries_summary
    plot_timeseries_summary(Out_all, hist_end_year, vars_final, out_dir=out_dir, show_plot=show_plot)
    
    return Out_all

def _validate_choice(value, allowed_list, name):
    """
    Validates a user choice against a list that may contain nested 
    lists from YAML anchors.
    """
    # 1. Internal flattener to handle nested lists from YAML
    def flatten(nested):
        flat = []
        for item in nested:
            if isinstance(item, list):
                flat.extend(flatten(item))
            else:
                flat.append(item)
        return flat

    # 2. Get the clean, flat list of allowed options
    flat_allowed = flatten(allowed_list)

    # 3. Perform the check
    if value not in flat_allowed:
        raise ValueError(
            f"Invalid {name}: '{value}'. "
            f"Currently supported in this version: {flat_allowed}"
        )