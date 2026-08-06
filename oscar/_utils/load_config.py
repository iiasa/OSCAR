import yaml
from oscar._io.paths import PACKAGE_ROOT

def load_variable_names(
    variable_path: str, section: str = "anthropogenic_emissions"
) -> list[str]:
    with open(variable_path, "r", encoding="utf-8-sig") as f:
        data = yaml.safe_load(f)
        if not data or section not in data:
            return []
        return list(data[section].keys())

def load_config():
    config_path = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "config.yaml"
    # 'utf-8-sig' ignores the hidden Windows BOM marker
    with open(config_path, "r", encoding="utf-8-sig") as f:
        config = yaml.safe_load(f)
        if config is None:
            raise ValueError(f"Config file is empty or invalid: {config_path}")

        variable_path = PACKAGE_ROOT / "oscar" / "_utils" / "_resources" / "variables.yaml"
        config["v_all"] = load_variable_names(variable_path)

        return config

