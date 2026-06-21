import yaml
from .._io.paths import PACKAGE_ROOT

def load_config():
    config_path = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "config.yaml"
    # 'utf-8-sig' ignores the hidden Windows BOM marker
    with open(config_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
        if data is None:
            raise ValueError(f"Config file is empty or invalid: {config_path}")
        return data