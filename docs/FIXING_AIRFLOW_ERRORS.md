# Fixing Airflow DAG Import Errors

## 🔴 Problem

You're seeing these errors in Airflow UI:
- `ImportError: cannot import name '_CloudAccessor'`
- `PythonVirtualenvOperator requires virtualenv`
- DAG Import Errors showing example DAGs from venv/lib/python3.11/site-packages/airflow/example_dags/

## 🎯 Root Cause

1. **Example DAGs are still loading** even though you set `load_examples = False`
2. **Missing virtualenv package** in your environment
3. **airflow.cfg might have duplicate [core] sections** or settings not applied correctly

## ✅ Complete Fix (Step-by-Step)

### Step 1: Stop All Airflow Processes

In **both Terminal 1 and Terminal 2**, press:
```
Ctrl + C
```

Wait for graceful shutdown. If processes don't stop, run:
```bash
pkill -f airflow
```

### Step 2: Run the Automated Fix Script

In WSL terminal:
```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
chmod +x fix_airflow_config.sh
./fix_airflow_config.sh
```

This script will:
- ✅ Backup your current airflow.cfg
- ✅ Set correct dags_folder path
- ✅ Disable example DAGs
- ✅ Install missing virtualenv package

### Step 3: Clean Cached Files

```bash
# Remove scheduler cache
rm -rf logs/*

# Remove database lock files
rm -f airflow.db-shm airflow.db-wal

# Recreate .gitkeep files
touch logs/.gitkeep
```

### Step 4: Verify Your DAG File Exists

```bash
ls -la dags/
cat dags/demand_forecasting_pipeline.py | head -20
```

You should see your `demand_forecasting_pipeline.py` file.

### Step 5: Restart Airflow Services

**Terminal 1 - Webserver:**
```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow webserver --port 8080
```

**Terminal 2 - Scheduler:**
```bash
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow scheduler
```

### Step 6: Verify in UI

1. Open http://localhost:8080
2. Login (admin/admin)
3. You should now see **ONLY** your `demand_forecasting_pipeline` DAG
4. No import errors should appear

---

## 🛠️ Manual Fix (If Script Doesn't Work)

### Option A: Edit airflow.cfg Manually

```bash
nano airflow.cfg
```

Find the `[core]` section (should be near the top, around line 20-30) and update:

```ini
[core]
dags_folder = /mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting/dags
load_examples = False
```

**Important**: 
- Make sure there's only ONE `[core]` section
- Don't add a duplicate `[core]` at the bottom
- Use the FULL absolute path starting with `/mnt/d/`

Save with `Ctrl+O`, `Enter`, then exit with `Ctrl+X`.

### Option B: Regenerate airflow.cfg

```bash
# Backup old config
mv airflow.cfg airflow.cfg.old

# Set environment variable BEFORE initializing
export AIRFLOW_HOME=$(pwd)
export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
export AIRFLOW__CORE__LOAD_EXAMPLES=False

# Reinitialize database (this creates new airflow.cfg)
airflow db init

# Recreate admin user
airflow users create \
    --username admin \
    --firstname Admin \
    --lastname User \
    --role Admin \
    --email admin@example.com \
    --password admin
```

### Option C: Install Missing Dependencies

```bash
source venv/bin/activate
pip install virtualenv
pip install 'universal-pathlib>=0.2.2'
```

---

## 🔍 Verification Commands

After applying the fix, verify everything is correct:

### Check Configuration
```bash
# View dags_folder setting
grep "^dags_folder" airflow.cfg

# View load_examples setting
grep "^load_examples" airflow.cfg

# Should output:
# dags_folder = /mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting/dags
# load_examples = False
```

### Check DAG Discovery
```bash
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)

# List all DAGs (should only show your DAG)
airflow dags list

# Should output:
# demand_forecasting_pipeline
```

### Check for Import Errors
```bash
# This should return nothing if all is well
airflow dags list-import-errors
```

---

## 🎯 Expected Result

After the fix, you should see:

✅ **In Airflow UI:**
- No "DAG Import Errors" banner
- Only 1 DAG visible: `demand_forecasting_pipeline`
- No example DAGs (tutorial_objectstorage, example_branch_operator, etc.)

✅ **In Terminal:**
```bash
$ airflow dags list
dag_id                        | filepath                                      | owner   | paused
==============================|===============================================|=========|========
demand_forecasting_pipeline   | demand_forecasting_pipeline.py                | airflow | True
```

---

## 🆘 Still Having Issues?

### Issue: "No DAGs found"

**Solution:**
```bash
# Check if DAG file exists
ls -la dags/demand_forecasting_pipeline.py

# Check if it's valid Python
python -m py_compile dags/demand_forecasting_pipeline.py

# Check for syntax errors
python dags/demand_forecasting_pipeline.py
```

### Issue: "Permission denied on dags folder"

**Solution:**
```bash
# Fix permissions
chmod -R 755 dags/
```

### Issue: "Still seeing example DAGs"

**Solution:**
```bash
# Make absolutely sure load_examples is False
sed -i 's/load_examples = True/load_examples = False/g' airflow.cfg

# Verify
grep load_examples airflow.cfg

# Clear browser cache and refresh
```

### Issue: "Configuration not taking effect"

**Solution:**
```bash
# Make sure AIRFLOW_HOME is set correctly
echo $AIRFLOW_HOME
# Should output: /mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting

# If not, set it:
export AIRFLOW_HOME=$(pwd)

# Verify airflow.cfg location
airflow config get-value core dags_folder
```

---

## 📋 Quick Checklist

Before restarting Airflow, verify:

- [ ] All Airflow processes stopped (`pkill -f airflow`)
- [ ] `airflow.cfg` has correct `dags_folder` path
- [ ] `airflow.cfg` has `load_examples = False`
- [ ] Only ONE `[core]` section in `airflow.cfg`
- [ ] `virtualenv` package installed (`pip list | grep virtualenv`)
- [ ] `AIRFLOW_HOME` environment variable set (`echo $AIRFLOW_HOME`)
- [ ] DAG file exists in `dags/` folder
- [ ] Cached files cleaned (`rm -rf logs/*`)

---

## 💡 Pro Tips

1. **Always set AIRFLOW_HOME** before running any airflow command:
   ```bash
   export AIRFLOW_HOME=$(pwd)
   ```

2. **Use environment variables** for configuration:
   ```bash
   export AIRFLOW__CORE__LOAD_EXAMPLES=False
   export AIRFLOW__CORE__DAGS_FOLDER=$(pwd)/dags
   ```

3. **Check scheduler logs** if DAG doesn't appear:
   ```bash
   tail -f logs/scheduler/latest/scheduler.log
   ```

4. **Refresh DAGs** without restarting:
   - In UI, click the refresh button (circular arrow icon)

---

## 🚀 Quick Fix Command (All-in-One)

If you want to fix everything in one go:

```bash
# Stop Airflow
pkill -f airflow

# Navigate to project
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"

# Run fix script
chmod +x fix_airflow_config.sh
./fix_airflow_config.sh

# Clean cache
rm -rf logs/*
rm -f airflow.db-shm airflow.db-wal
touch logs/.gitkeep

# Restart (Terminal 1)
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow webserver --port 8080

# Restart (Terminal 2 - in new terminal)
cd "/mnt/d/My Apps/Airflow Demand Forecast Project/airflow-demand-forecasting"
source venv/bin/activate
export AIRFLOW_HOME=$(pwd)
airflow scheduler
```

---

**After following these steps, your Airflow UI should be clean with only your demand forecasting DAG visible!** 🎉
