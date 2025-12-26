# Multi-Model Forecasting - Quick Start Guide

## 🚀 Quick Start (5 Minutes)

Follow these steps to get your multi-model forecasting system running:

### Step 1: Install Dependencies (2 minutes)

```bash
# Navigate to project directory
cd "d:\My Apps\Airflow Demand Forecast Project\airflow-demand-forecasting"

# Install base requirements
pip install -r requirements/base.txt

# Install Prophet
pip install -r requirements/prophet.txt

# Install SARIMA
pip install -r requirements/sarima.txt

# Install Ensemble
pip install -r requirements/ensemble.txt

# Optional: Install DeepAR (requires gluonts + mxnet)
# pip install -r requirements/deepar.txt
```

### Step 2: Verify Installation (30 seconds)

```bash
# Test configuration loading
python scripts/common/config_loader.py

# Expected output:
# ✅ PROPHET config loaded successfully
# ✅ SARIMA config loaded successfully
# ✅ DEEPAR config loaded successfully
# ✅ ENSEMBLE config loaded successfully
```

### Step 3: Run Unit Tests (1 minute)

```bash
# Run all tests
pytest tests/ -v

# Expected: All tests should pass
```

### Step 4: Prepare Your Data (1 minute)

Your data should have these columns:
- `ds` or `date` - Date column
- `sales_quantity` or `y` - Target variable
- `product_id` - Product identifier (for SARIMA)
- `revenue` - External feature (for DeepAR, optional)

Example CSV format:
```csv
ds,product_id,sales_quantity,revenue
2024-01-01,PROD_001,100,1000
2024-01-02,PROD_001,110,1100
2024-01-03,PROD_001,105,1050
```

Place your data in:
```
data/processed/cleaned_data.csv
```

### Step 5: Train Models (30 seconds per model)

#### Train Prophet:
```bash
python scripts/prophet/model_training.py --data data/processed/cleaned_data.csv
```

#### Train SARIMA (per-product):
```bash
python scripts/sarima/model_training.py --data data/processed/cleaned_data.csv
```

#### Optional: Train DeepAR:
```bash
python scripts/deepar/model_training.py --data data/processed/cleaned_data.csv
```

### Step 6: Generate Predictions

#### Prophet Forecast:
```bash
python scripts/prophet/prediction.py
```

#### Create Ensemble Forecast:
```bash
python scripts/ensemble/model_combiner.py \
  --prophet data/predictions/prophet/forecast_YYYYMMDD.csv \
  --sarima data/predictions/sarima/forecast_YYYYMMDD.csv
```

---

## 📊 Expected Results

After running all steps, you should have:

1. **Trained Models:**
   - `models/prophet/prophet_vYYYYMMDD_*.pkl`
   - `models/sarima/PROD_001/sarima_PROD_001_*.pkl`
   - `models/sarima/PROD_002/sarima_PROD_002_*.pkl`
   - etc.

2. **Predictions:**
   - `data/predictions/prophet/forecast_YYYYMMDD.csv`
   - `data/predictions/sarima/forecast_YYYYMMDD.csv`
   - `data/predictions/ensemble/forecast_YYYYMMDD.csv`

3. **Metadata:**
   - `models/prophet/*_metadata.json`
   - `models/sarima/PROD_001/*_metadata.json`
   - etc.

---

## 🎯 Validation Checklist

- [ ] All dependencies installed successfully
- [ ] Configuration files loaded without errors
- [ ] Unit tests pass
- [ ] Prophet model trains successfully
- [ ] SARIMA models train for all products
- [ ] Predictions generated successfully
- [ ] Ensemble forecast created
- [ ] Validation MAPE < 20% for individual models
- [ ] Ensemble MAPE < individual model MAPE

---

## 🔧 Troubleshooting

### Issue: Import errors
**Solution:** Ensure you're in the project root directory and Python can find the scripts:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### Issue: "No module named 'prophet'"
**Solution:** Install Prophet:
```bash
pip install prophet cmdstanpy
```

### Issue: "No module named 'pmdarima'"
**Solution:** Install pmdarima for auto-ARIMA:
```bash
pip install pmdarima
```

### Issue: SARIMA training very slow
**Solution:** Enable parallel training in `config/sarima_config.yaml`:
```yaml
training:
  parallel_training:
    enabled: true
    n_jobs: -1
```

### Issue: DeepAR not working
**Solution:** DeepAR is a template. For full functionality:
```bash
pip install gluonts mxnet
```

---

## 📚 Next Steps

Once basic setup is complete:

1. **Optimize Hyperparameters:**
   - Edit `config/prophet_config.yaml`
   - Edit `config/sarima_config.yaml`
   - Retrain models

2. **Tune Ensemble Weights:**
   - Try different weight strategies in `config/ensemble_config.yaml`
   - Compare auto vs manual weights

3. **Deploy to Airflow:**
   - Copy DAGs to Airflow DAGs folder
   - Configure Airflow connections
   - Test DAG execution

4. **Monitor Performance:**
   - Track validation metrics
   - Compare ensemble vs individual models
   - Retrain periodically

---

## 🎓 Learning Resources

- **Full Documentation:** `README_MULTI_MODEL.md`
- **Implementation Details:** `IMPLEMENTATION_SUMMARY.md`
- **Configuration Guide:** See `config/*.yaml` files
- **Code Examples:** See `scripts/*/model_training.py` files

---

## 💡 Pro Tips

1. **Start with Prophet:** It's the fastest and easiest to get working
2. **Use Small Dataset First:** Test with 100-200 records before full dataset
3. **Check Validation Metrics:** MAPE should be < 20% for good performance
4. **Enable Logging:** Set `logging.level: DEBUG` in configs for troubleshooting
5. **Version Control:** Commit configs before making changes

---

## ✅ Success Criteria

You're ready for production when:

- ✅ All models train without errors
- ✅ Validation MAPE < 15% for ensemble
- ✅ Ensemble outperforms individual models
- ✅ Predictions save successfully
- ✅ Metadata tracks all model versions
- ✅ Tests pass with > 80% coverage

---

## 🎉 You're Ready!

Your multi-model forecasting system is now set up and ready to use!

**Questions?** Check `README_MULTI_MODEL.md` or review the code documentation.

**Happy Forecasting! 📈**
