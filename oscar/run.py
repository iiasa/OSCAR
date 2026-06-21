# oscar/run.py
from oscar._utils.help import show_info as _info
from oscar._io._download import ensure_configured_library, ensure_customized_library

def info(mode=None):
    """Entry point for terminal-based help."""
    return _info(mode)


def download(mode="customized", hist_type=None, region=None):
    """
    Public data download helper.

    Args:
        mode (str): "customized" (default) or "configured".
        hist_type (str): Required only for configured mode.
        region (str): Required only for configured mode.

    Returns:
        Path to the downloaded/existing local library.
    """
    if mode == "customized":
        return ensure_customized_library()

    if mode == "configured":
        if not hist_type or not region:
            raise ValueError(
                "Configured mode download requires both 'hist_type' and 'region'."
            )
        return ensure_configured_library(hist_type, region)

    raise ValueError("Unknown mode. Supported values are: 'customized', 'configured'.")

import sys
from oscar._io.paths import get_user_data_dir, PACKAGE_ROOT

def run(mode="standard", **kwargs):
    """
    Main entry point for OSCAR simulations.
    """
    
    # 1. MODE: STANDARD (Instant verification using internal bootstrap)
    if mode == "standard":
        from oscar._workflows import standard_run
        return standard_run.run_standard(**kwargs)

    # 2. DATA CHECK (For scientific modes: configured, customized, advanced)
    # This checks for a saved path OR a manually provided 'data_dir' argument.
    data_root = get_user_data_dir()
    
    if data_root is None:
        # --- THE WARM WELCOME GUIDE ---
        width = 85
        print("\n" + "="*width)
        print(f"{' WELCOME TO OSCAR ':^85}")
        print("="*width)
        print("You have selected a mode that requires a designated home for the data library.")
        print("To start your scientific research, please initialize your data folder once.")
        
        print("\nACTION: Run one of these commands in your terminal:")
        print(f"  {'# To use the default [Project Root]/data:':<50}")
        print("  python -c \"import oscar; oscar.set_data_dir()\"")
        
        print(f"\n  {'# To use a custom external drive:':<50}")
        print("  python -c \"import oscar; oscar.set_data_dir('/path/to/your/drive')\"")
        
        print("\nOnce initialized, OSCAR will automatically manage and download required data.")
        print("="*width + "\n")
        
        # Exit cleanly without a technical traceback
        sys.exit(1)

    # 3. SCIENTIFIC WORKFLOWS (Now guaranteed to have a data_root)
    if mode == "configured":
        from oscar._workflows import configured_runs
        return configured_runs.run_configured(**kwargs)

    if mode == "customized":
        from oscar._workflows import customized_runs
        return customized_runs.run_customized(**kwargs)

    if mode == "advanced":
        from oscar._workflows import advanced_runs
        return advanced_runs.run_advanced(**kwargs)
        
    raise ValueError(f"Unknown mode: {mode}")