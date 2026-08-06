# OSCAR 🌍

We are excited to announce the release of **OSCAR v4**! OSCAR is an open-source, reduced-complexity Earth system model (ESM) designed to probabilistically emulate the coupled climate–carbon–chemistry system. This latest version introduces enhanced numerical stability, modular submodels, and recalibration using AR6, CMIP6, and TRENDY datasets, among other things.

> [!WARNING]
> ### ⚠️ Beta Version - Active Development
> This package is currently in **Beta (v4-beta1)**. We are actively refining features and documentation.
>
> We expect a stable v4.0 by the end of 2026.
> 
> **Action Required:** Expect frequent updates in the coming months; please keep your package up to date.
> 
> **Support & Feedback:**
> - **Technical Issues:** For bug reports or feature suggestions, please open an issue on the [Issue Board](https://github.com/iiasa/OSCAR/issues).
> - **Scientific Collaboration:** For research inquiries, contact the OSCAR model team at [OSCAR_model@iiasa.ac.at](mailto:OSCAR_model@iiasa.ac.at).



## 📖 Scientific Background

OSCAR v4 bridges comprehensive ESMs and simpler reduced-complexity approaches, providing rapid, policy-relevant, probabilistic climate projections. 

**Key Improvements:**
- User-oriented functionalities.
- Modularized code for independent submodel execution.
- Robust Monte Carlo & Latin hypercube sampling for improved probabilistic ensembles.
- Benchmarked via the Reduced Complexity Model Intercomparison Project (RCMIP) phase 3.

**References:**

**v4.0 |** Gasser et al., *in preparation*.

**v3.1 |** Gasser, T., L. Crepin, Y. Quilcaille, R. A. Houghton, P. Ciais & M. Obersteiner. "Historical CO<sub>2</sub> emissions from land-use and land-cover change and their uncertainty." *Biogeosciences* 17: 4075–4101 (2020). [doi:10.5194/bg-17-4075-2020](https://doi.org/doi:10.5194/bg-17-4075-2020)

**v2.2 |** Gasser, T., P. Ciais, O. Boucher, Y. Quilcaille, M. Tortora, L. Bopp & D. Hauglustaine. "The compact Earth system model OSCAR v2.2: description and first results." *Geoscientific Model Development* 10: 271-319 (2017). [doi:10.5194/gmd-10-271-2017](https://doi.org/doi:10.5194/gmd-10-271-2017)

---

## 🚀 Installation

OSCAR can be installed via `pip` or `uv`. We recommend **uv** for a faster and cleaner environment setup.

### Option 1: Using uv (Recommended)
*Requires [uv](https://astral.sh/uv).*
```bash
# Create a virtual environment and install OSCAR
uv venv
uv pip install git+https://github.com/iiasa/OSCAR.git

# Run the verification model
uv run oscar run
```

### Option 2: Using standard pip
```bash
# Install directly from GitHub
pip install git+https://github.com/iiasa/OSCAR.git

# Run the verification model
oscar run
```

---

## 🔎 Discovery, Help & Run Modes

OSCAR is self-documenting and supports multiple run levels.  
You can both **inspect** a mode and **execute** it from the Terminal or Python.

| Mode | Purpose | Inspect (Terminal) | Inspect (Python) | Run (Terminal) | Run (Python) | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **General** | Package overview & guidance | `oscar` | `oscar.info()` | — | — | ✅ Available |
| **Standard** | Instant verification simulation | `oscar info standard` | `oscar.info("standard")` | `oscar run` | `oscar.run()` | ✅ Available |
| **Configured** | Official scenario library runs | `oscar info configured` | `oscar.info("configured")` | `oscar run -m configured` | `oscar.run(mode="configured")` | ✅ Available |
| **Customized** | User-defined research workflows | — | `oscar.info("customized")` | — | `oscar.run(mode="customized")` | ✅ Available |
| **Advanced** | Core model development | — | — | — | `oscar.run(mode="advanced")` | 🚧 In Development |

> **Note:** Scientific modes (Configured and above) require access to a large data library.  
> The model will guide you through a one-time directory initialization on first use.

---

## 🛠 Usage Examples

### Terminal (Standard Mode)
Run the 2024-2100 verification simulation with a single command:
```bash
oscar run
```

### Python (Configured Mode)
Run a specific CMIP7 regional projection for multiple SSP scenarios:
```python
import oscar

# Run official scenarios for the 5 RCP regions
oscar.run(
    mode="configured", 
    scenario=["scen7-VL","scen7-H"], 
    variables=["D_Tg"]
)
```
### Python (Customized Mode)
Run user defined scenarios

**Preparation:** 

0. Download Templates: Look for `setting_template.yaml`

1. Create Project Folder : `oscar.create_project('my-project')`
    - (optional) put user forcing in this project folder (if empty, model will run with baseline scenarios)
    - modify the setting file: `setting_my-experiment.yaml`

2. Run Simulation with: 
```python
import oscar

# Run user-defined scenarios
oscar.run(
    mode="customized", 
    project='my-project', 
    experiment='my-experiment'
)
```
**Note:** run `oscar.info('customized')` for detailed information.

---

## 🗑 Uninstallation

To remove OSCAR and its associated files:

| Target | Command / Action |
| :--- | :--- |
| **Package (uv)** | Delete the `.venv` folder and the project directory. |
| **Package (pip)** | Run `pip uninstall oscar`. |
| **Data Library** | Manually delete the data folder you initialized during setup. |

