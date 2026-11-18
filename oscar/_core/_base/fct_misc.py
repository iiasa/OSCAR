import numpy as np
import xarray as xr
from scipy.stats import theilslopes


##################################################
##################################################

## helper for linear fits on xarrays
def xr_linfit(y, x, dim='year', method='ols', intercept=None):
    '''
    Linear fit y = intercept + slope * x along 'dim', independently for all other dims.
    - y, x: xarray DataArrays sharing 'dim'
    - method: 'ols' or 'theilsen'
    - intercept: if not None, forces the fit through this intercept (scalar or xarray without 'dim'; NaN = free fit)
    Returns a Dataset with 'slope' and 'intercept'.
    '''
    def _fit(y, x, c, forced):
        if not (np.all(np.isfinite(y)) and np.all(np.isfinite(x))) or (forced and not np.isfinite(c)):
            return np.nan, np.nan
        try:
            if not forced:
                if method == 'theilsen':
                    s, c = theilslopes(y, x)[:2]
                else:
                    s, c = np.polyfit(x, y, 1)
            elif method == 'theilsen':
                ok = x != 0
                s = np.median((y[ok] - c) / x[ok])
            else:
                s = np.sum(x * (y - c)) / np.sum(x * x)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            return np.nan, np.nan
        return s, c
    forced = intercept is not None
    c = xr.DataArray(np.nan) if intercept is None else intercept
    slope, intercept = xr.apply_ufunc(_fit, y, x, c, forced, input_core_dims=[[dim], [dim], [], []],
                                      output_core_dims=[[], []], vectorize=True,
                                      dask='parallelized', output_dtypes=[float, float],
                                      dask_gufunc_kwargs={'allow_rechunk': True})
    return xr.Dataset({'slope': slope, 'intercept': intercept})

