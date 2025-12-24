#!/bin/bash
# Safe quick fix for Airflow example DAG import errors
# Author: Clean, production-style fix

set -e

PROJECT_DIR="$(pwd)"
CFG_FILE="$PROJECT_DIR/airflow.cfg"

echo "🔧 Stopping Airflow processes..."
pkill -f airflow || true
pkill -f gunicorn || true
sleep 2

echo "📄 Using airflow.cfg at:"
echo "   $CFG_FILE"
echo ""

if [ ! -f "$CFG_FILE" ]; then
  echo "❌ airflow.cfg not found. Aborting."
  exit 1
fi

echo "🧠 Ensuring correct Airflow core configuration..."

# Check if [core] exists
if ! grep -q "^\[core\]" "$CFG_FILE"; then
  echo "⚠️  [core] section not found. Adding it."
  echo "" >> "$CFG_FILE"
  echo "[core]" >> "$CFG_FILE"
fi

# Ensure dags_folder is set correctly (append if missing)
if grep -q "^dags_folder" "$CFG_FILE"; then
  sed -i "s|^dags_folder *=.*|dags_folder = $PROJECT_DIR/dags|" "$CFG_FILE"
else
  sed -i "/^\[core\]/a dags_folder = $PROJECT_DIR/dags" "$CFG_FILE"
fi

# Ensure load_examples = False
if grep -q "^load_examples" "$CFG_FILE"; then
  sed -i "s/^load_examples *=.*/load_examples = False/" "$CFG_FILE"
else
  sed -i "/^\[core\]/a load_examples = False" "$CFG_FILE"
fi

echo ""
echo "✅ Airflow core configuration updated:"
echo "-----------------------------------"
grep -E "^\[core\]|^dags_folder|^load_examples" "$CFG_FILE"
echo "-----------------------------------"

echo ""
echo "🚀 Fix complete."

echo ""
echo "Now restart Airflow:"
echo ""
echo "Terminal 1:"
echo "  source venv/bin/activate"
echo "  export AIRFLOW_HOME=\$(pwd)"
echo "  airflow webserver --port 8080"
echo ""
echo "Terminal 2:"
echo "  source venv/bin/activate"
echo "  export AIRFLOW_HOME=\$(pwd)"
echo "  airflow scheduler"
echo ""
