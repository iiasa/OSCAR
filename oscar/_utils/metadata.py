"""
OSCAR Metadata Manager
Location: oscar/_utils/metadata.py

This module handles the registration of scientific metadata (units, long names)
onto model objects. It serves as the bridge between raw model output and
international reporting standards (CF-Conventions).
"""

import yaml
from oscar._io.paths import PACKAGE_ROOT

def get_oscar_version():
    """
    Reads the dynamic version from the VERSION file in the package root.
    """
    version_path = PACKAGE_ROOT / "VERSION"
    if version_path.exists():
        with open(version_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "unknown"

def load_var_registry():
    """
    Loads the official variable definitions from the resources folder.
    Uses utf-8-sig to safely handle Windows/Network drive encodings.
    """
    path = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "variables.yaml"
    
    if not path.exists():
        raise FileNotFoundError(f"Metadata Registry missing: {path}")
        
    with open(path, "r", encoding="utf-8-sig") as f:
        # Load the dictionary under the 'variables' key
        data = yaml.safe_load(f)
        return data.get('variables', {})

def apply_variable_metadata(ds):
    registry = load_var_registry()
    version = get_oscar_version() # Fetch version
    
    # Update global tags
    ds.attrs['model'] = f"OSCAR"
    ds.attrs['version'] = version
    
    for var in ds.data_vars:
        if var in registry:
            meta = registry[var]
            updates = {
                'units': meta.get('unit', 'n/a'),
                'long_name': meta.get('long_name', var),
                'sci_name': meta.get('sci_name', var)
            }
            ds[var].attrs.update(updates)
            
    return ds

# --- Future-Proofing placeholders ---

def apply_parameter_metadata(par_ds):
    """
    (Placeholder) To be implemented in later versions for 
    labeling Monte Carlo parameter sets.
    """
    pass