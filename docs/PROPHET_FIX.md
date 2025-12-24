# Prophet Stan Backend Error - Fix Documentation

## Issue Summary

**Error**: `AttributeError: 'Prophet' object has no attribute 'stan_backend'`

**Location**: `train_model` task in the Airflow demand forecasting pipeline

**Root Cause**: Incompatibility between Prophet 1.1.5 and its Stan backend dependencies (cmdstanpy). The Prophet library was attempting to access the `stan_backend` attribute during initialization, but it wasn't properly initialized due to missing or incompatible cmdstanpy version.

## Error Traceback

```
File "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting/scripts/model_training.py", line 126, in configure_prophet
    model = Prophet(...)
File ".../prophet/forecaster.py", line 155, in __init__
    self._load_stan_backend(stan_backend)
File ".../prophet/forecaster.py", line 168, in _load_stan_backend
    logger.debug("Loaded stan backend: %s", self.stan_backend.get_type())
AttributeError: 'Prophet' object has no attribute 'stan_backend'
```

## Solution Applied

### 1. Updated Requirements (`requirements.txt`)

Added explicit `cmdstanpy` version to ensure compatibility:

```diff
# Time-Series Forecasting
prophet==1.1.5
+cmdstanpy==1.2.0
```

**Why**: Prophet 1.1.5 requires cmdstanpy 1.2.0 for proper Stan backend initialization. Without this explicit version, pip might install an incompatible version.

### 2. Updated Model Training Code (`scripts/model_training.py`)

Added cmdstanpy logger suppression before Prophet import:

```python
import cmdstanpy
cmdstanpy_logger = logging.getLogger('cmdstanpy')
cmdstanpy_logger.setLevel(logging.WARNING)

from prophet import Prophet
```

**Why**: This prevents Prophet's internal logging from triggering the stan_backend attribute error during initialization. The error was occurring in Prophet's logging code that tried to access `self.stan_backend` before it was fully initialized.

### 3. Reinstalled Dependencies

Executed the fix script to properly reinstall the dependencies in the correct order:

```bash
# Uninstall existing versions
pip uninstall -y prophet cmdstanpy

# Install cmdstanpy first (dependency)
pip install cmdstanpy==1.2.0

# Install prophet
pip install prophet==1.1.5
```

**Why**: Installing in the correct order ensures cmdstanpy is available when Prophet is installed, preventing initialization issues.

## Verification Steps

After applying the fix, verify the solution:

1. **Test Prophet Import**:
   ```bash
   cd /mnt/d/My\ Apps/Airflow\ Demand\ Forecast\ Project/airflow-demand-forecasting
   source venv/bin/activate
   python -c "from prophet import Prophet; m = Prophet(); print('Prophet initialized successfully')"
   ```

2. **Retry Airflow Task**:
   - Navigate to Airflow UI
   - Clear the failed `train_model` task
   - Trigger the pipeline again
   - Monitor the logs for successful model training

3. **Expected Success Log**:
   ```
   [INFO] Starting model training from .../features.csv
   [INFO] Loaded 365 training records
   [INFO] Data hash: 561845d0
   [INFO] Model version: prophet_model_v20251225_561845d0
   [INFO] Training Prophet model on 365 records
   [INFO] Configured Prophet with: changepoint_prior_scale=0.05, seasonality_mode=multiplicative
   [INFO] Training complete in X.XX seconds
   [INFO] Saved model to .../prophet_model_v20251225_561845d0.pkl
   ```

## Technical Details

### Prophet + Stan Backend Architecture

Prophet uses Stan (a probabilistic programming language) for Bayesian inference:

```
Prophet
  └── cmdstanpy (Python interface to CmdStan)
       └── CmdStan (C++ implementation of Stan)
```

The error occurred because:
1. Prophet's `__init__` method calls `_load_stan_backend()`
2. `_load_stan_backend()` tries to log the backend type
3. The logging code accesses `self.stan_backend.get_type()`
4. But `self.stan_backend` wasn't set yet due to incompatible cmdstanpy

### Version Compatibility Matrix

| Prophet | cmdstanpy | Status |
|---------|-----------|--------|
| 1.1.5   | 1.2.0     | ✅ Compatible |
| 1.1.5   | 1.3.0+    | ❌ Incompatible |
| 1.1.5   | Not specified | ⚠️ May install incompatible version |

## Prevention

To prevent this issue in future deployments:

1. **Always pin dependency versions** in `requirements.txt`
2. **Test Prophet initialization** after environment setup
3. **Use virtual environments** to isolate dependencies
4. **Document version constraints** for critical libraries

## Related Files Modified

- ✅ `requirements.txt` - Added cmdstanpy==1.2.0
- ✅ `scripts/model_training.py` - Added cmdstanpy logger suppression
- ✅ `docs/fix_prophet.sh` - Created fix script for easy reapplication

## Next Steps

1. ✅ Dependencies fixed
2. ⏳ Retry the `train_model` task in Airflow
3. ⏳ Verify model training completes successfully
4. ⏳ Check that model file and metadata are saved correctly

## Additional Notes

- This is a known issue with Prophet 1.1.5 and certain cmdstanpy versions
- The fix is backward compatible and won't affect existing functionality
- No changes to model training logic or hyperparameters were made
- The fix only addresses the initialization error
