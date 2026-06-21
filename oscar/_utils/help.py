"""
OSCAR Help Utility
Location: oscar/_utils/help.py

Provides contextual information about run modes, official scientific 
scenarios, and regional configurations.
"""
from .load_config import load_config

def show_info(mode=None):
    """Router for the OSCAR information system."""
    cfg_full = load_config()
    
    # Normalize mode string
    m = str(mode).lower() if mode else None

    if m is None or m == 'none':
        _print_general()
    elif m == 'standard':
        _print_standard(cfg_full)
    elif m == 'configured':
        _print_configured(cfg_full)
    elif m == 'customized':
        _print_customized(cfg_full)
    elif m == 'advanced':
        _print_advanced()
    else:
        print(f"\n[!] Unknown mode: '{mode}'")
        print("Available modes are: 'standard', 'configured', 'customized', 'advanced'")

def _print_general():
    width = 90
    print("\n" + "="*width)
    print(f"{'OSCAR MODEL - GENERAL OVERVIEW':^90}")
    print("="*width)
    print("A reduced-complexity Earth system model for climate research.")
    
    print("\nAvailable Run Modes:")
    print(f"  {'standard':<12} : Fast verification (no setup). → Command: oscar run")
    print(f"  {'configured':<12} : Official CMIP runs.           → Info:    oscar info configured")
    print(f"  {'customized':<12} : User-defined research.        → Info:    oscar info customized")
    print(f"  {'advanced':<12} : [DEV] Model sub-modules.       (Status: Internal Use Only)")
    print("="*width + "\n")

def _print_standard(cfg_full):
    """Displays information for the Standard verification mode."""
    cfg = cfg_full['standard_mode']
    width = 90
    print("\n" + "="*width)
    print(f"{'MODE: STANDARD (Verification)':^90}")
    print("="*width)
    print("Goal:        Fast proof-of-concept run for the model installation.")
    print(f"Data source: Internal package bootstrap ({cfg['nMC']} members).")
    print(f"Timeline:    {cfg['run_range'][0]} to {cfg['run_range'][1]}.")
    print(f"History:     {cfg['hist_type']}")
    print(f"Region:      Fixed ({cfg['region']}).")
    
    print("\nExample Commands:")
    print("  [Terminal] : oscar run")
    print("  [Python]   : import oscar; oscar.run()")
    print("="*width + "\n")

def _print_configured(cfg_full):
    """Displays official scientific library options."""
    cfg = cfg_full['configured_mode']
    width = 95
    print("\n" + "="*width)
    print(f"{'MODE: CONFIGURED (Scientific Library)':^95}")
    print("="*width)
    print("Official scientific experiments using curated forcing and parameter libraries.")
    
    print("\nAvailable Options :")
    print(f"  {'Histories':<12} : {', '.join(cfg['allowed_hist'])}")
    print(f"  {'Regions':<12} : {', '.join(cfg['allowed_regions'])}")
    print(f"  {'Scenarios':<12} : {', '.join(cfg['allowed_scenarios'])}")
    
    # Metadata lookup for descriptions
    from .metadata import load_var_registry
    var_meta = load_var_registry()
    
    # --- FIX: FLATTEN THE LIST ---
    # Since YAML anchors [*a, *b] can sometimes result in nested structures 
    # depending on the loader, we ensure we have a flat list of strings.
    raw_vars = cfg['allowed_variables']
    flat_vars = []
    for item in raw_vars:
        if isinstance(item, list):
            flat_vars.extend(item)
        else:
            flat_vars.append(item)
    # -----------------------------

    print(f"  {'Output Vars':<12} :")
    for var in flat_vars:
        # Now 'var' is guaranteed to be a string (hashable)
        desc = var_meta.get(var, {}).get('long_name', 'Official variable')
        print(f"{'':<15}- {var:<10}: {desc}")

    print(f"  {'MC Ensemble':<12} : {cfg['official_nMC']} members (Pre-validated)")

    print("\nExample Commands:")
    print("  [Terminal] : oscar run -m configured -s scen7-VL -r RCP_5reg -v D_Tg")
    print("  [Python]   : oscar.run(mode='configured', scenario='scen7-VL', variables=['D_Tg'])")
    print("="*width + "\n")

def _print_customized(cfg_full):
    """Displays instructions for research mode (Tier 2)."""
    cfg = cfg_full['customized_mode']
    width = 95
    print("\n" + "="*width)
    print(f"{'MODE: CUSTOMIZED (User Research)':^95}")
    print("="*width)
    print("Run OSCAR with custom experimental forcing and extended horizons.")
    
    print("\n[ SCIENTIFIC BOUNDARIES ]")
    y_min = cfg['allowed_years']['min']
    y_max = cfg['allowed_years']['max']
    print(f"  {'Timeline':<18}: {y_min} to {y_max}")
    print(f"  {'Supported Hist':<18}: {', '.join(cfg['allowed_hist'])}")
    print(f"  {'Ref Baselines':<18}: {', '.join(cfg['allowed_baselines'])}")
    print(f"  {'Output Regions':<18}: {', '.join(cfg['allowed_regions'])}")

    print("\n[ OUTPUT THEMES ]")
    themes = cfg['output_themes']
    for theme, raw_variables in themes.items():
        if isinstance(raw_variables, list):
            # --- FIX: FLATTEN THE THEME LIST ---
            flat_theme_vars = []
            for item in raw_variables:
                if isinstance(item, list):
                    flat_theme_vars.extend(item)
                else:
                    flat_theme_vars.append(item)
            # -----------------------------------
            var_str = ", ".join(flat_theme_vars)
        else:
            var_str = "(User-defined selection)"
        print(f"  - {theme:<15}: {var_str}")

    print("\n[ SETTINGS / WORKFLOW ]")
    for key, val in cfg['settings'].items():
        print(f"  {key:<20}: {val}/False")

    print("\n[ USAGE - FOLLOW THESE STEPS ]")
    print("  0. Download Templates : oscar.download(mode='customized')")
    print("  1. Create Project Folder : oscar.create_project('my-project')")
    print("  2. Edit setting file in the created folder (e.g., settings_my-experiment.yaml)")
    print("  3. Run Simulation        : oscar.run(mode='customized', project='my-project', experiment='my-experiment')")
    print("="*width + "\n")

def _print_advanced():
    """Information for Advanced development mode."""
    width = 90
    print("\n" + "="*width)
    print(f"{'MODE: ADVANCED (Model Development)':^90}")
    print("="*width)
    print("Direct control over model sub-modules and calibration internals.")
    print("\n( ! ) STATUS: Internal development only.")
    print("\nFuture Capabilities:")
    print("  - Selective Execution (Land only, Climate only)")
    print("  - Custom Calibration & Numerical Tuning")
    print("="*width + "\n")