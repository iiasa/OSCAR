"""
OSCAR Data Downloader - Configured Mode specialization
Handles the retrieval of official pre-compiled CMIP6/7 regional libraries.
"""
import zipfile
import requests
from oscar._io.paths import get_paths
from oscar._utils.load_config import load_config

def ensure_configured_library(hist_type, region):
    """
    Checks for the existence of a specific regional library.
    If missing, downloads and extracts the official bundle from Zenodo.
    """
    # 1. Resolve local path: data/configured/CMIP6/RCP_5reg/
    target_dir = get_paths()["configured_library"] / hist_type / region
    
    # Representative file check to see if library is already there
    if (target_dir / "forcing_hist.nc").exists():
        return target_dir

    # 2. If not found, prepare for download
    full_cfg = load_config()
    record_id = full_cfg['metadata']['run_mode']['zenodo_id']
    zip_filename = f"OSCAR_lib_tier1_configured_{hist_type}_{region}.zip"
    url = f"{full_cfg['metadata']['zenodo_base_url']}{record_id}/files/{zip_filename}/content"
    
    # Path to temporarily store the zip during download
    zip_temp_path = target_dir.parent / zip_filename
    zip_temp_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n[OSCAR] Official library for '{region}' ({hist_type}) not found locally.")
    print(f"[OSCAR] Please wait, fetching remote bundle from Zenodo...")
    
    try:
        # 3. Stream the download
        response = requests.get(url, stream=True)
        response.raise_for_status() # Ensure the Zenodo link is valid
        
        with open(zip_temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # 4. Extraction
        # Because the Bundler script included the region folder in the zip,
        # we extract it into the CMIP6 directory.
        print(f"[OSCAR] Extracting: {zip_filename}")
        with zipfile.ZipFile(zip_temp_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir.parent)

        print(f"[OSCAR] Setup complete. Library ready at {target_dir}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to download scientific library for {region}.\n"
            f"Please check your internet connection or URL: {url}\n"
            f"Error: {e}"
        )
    finally:
        # Cleanup temporary zip
        if zip_temp_path.exists():
            zip_temp_path.unlink()

    return target_dir


def ensure_customized_library():
    """Ensures the full Customized library is present locally.

    If missing, downloads OSCAR_lib_tier2_customized.zip from the configured
    Zenodo record and extracts it into
    {data_root}/library/run_mode/tier2_customized/.
    """
    target_dir = get_paths()["customized_library"]
    if target_dir is None:
        raise ValueError(
            "No user data directory configured. "
            "Please run oscar.set_data_dir('/your/path') first."
        )

    marker_file = target_dir / "templates" / "settings_template.yaml"
    if marker_file.exists():
        return target_dir

    full_cfg = load_config()
    metadata = full_cfg["metadata"]
    record_id = metadata["run_mode"]["zenodo_id"]
    zenodo_base_url = metadata["zenodo_base_url"]

    if not record_id:
        raise RuntimeError(
            "Customized library Zenodo record is not configured in config.yaml "
            "(metadata.customized.zenodo_id is empty)."
        )

    record_api_url = f"{zenodo_base_url}{record_id}"

    print("\n[OSCAR] Customized library not found locally.")
    print("[OSCAR] Please wait, fetching customized bundle from Zenodo...")

    zip_temp_path = None
    try:
        record_resp = requests.get(record_api_url, timeout=60)
        record_resp.raise_for_status()
        record_data = record_resp.json()

        target_filename = "OSCAR_lib_tier2_customized.zip"
        files = record_data.get("files", [])
        matching_files = [f for f in files if f.get("key") == target_filename]

        if not matching_files:
            raise RuntimeError(
                f"Expected file '{target_filename}' in Zenodo record "
                f"{record_id}, but it was not found."
            )

        zip_info = matching_files[0]
        zip_name = zip_info.get("key", target_filename)
        zip_url = zip_info.get("links", {}).get("self")
        if not zip_url:
            raise RuntimeError(
                f"Could not find a downloadable link for {target_filename} in "
                f"Zenodo record {record_id}."
            )

        target_dir.mkdir(parents=True, exist_ok=True)
        zip_temp_path = target_dir / zip_name

        response = requests.get(zip_url, stream=True, timeout=120)
        response.raise_for_status()
        with open(zip_temp_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"[OSCAR] Extracting: {zip_name}")
        with zipfile.ZipFile(zip_temp_path, "r") as zip_ref:
            zip_ref.extractall(target_dir)

        if not marker_file.exists():
            raise RuntimeError(
                "Customized bundle was extracted, but expected file "
                f"was not found: {marker_file}"
            )

        print(
            f"[OSCAR] Setup complete. Customized library ready at {target_dir}"
        )

    except Exception as e:
        raise RuntimeError(
            "Failed to download customized scientific library.\n"
            f"Zenodo record: {record_api_url}\n"
            f"Error: {e}"
        )
    finally:
        if zip_temp_path is not None and zip_temp_path.exists():
            zip_temp_path.unlink()

    return target_dir