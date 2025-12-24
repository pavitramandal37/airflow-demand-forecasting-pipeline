#!/bin/bash
# =============================================================================
# Fix Airflow Configuration - Remove Example DAGs and Set Correct DAGs Folder
# =============================================================================

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Fixing Airflow Configuration${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# Get the current directory (should be the project root)
PROJECT_DIR=$(pwd)
CONFIG_FILE="$PROJECT_DIR/airflow.cfg"

echo -e "${YELLOW}[1/5]${NC} Backing up current airflow.cfg..."
cp "$CONFIG_FILE" "$CONFIG_FILE.backup.$(date +%Y%m%d_%H%M%S)"
echo "  ✓ Backup created"
echo ""

echo -e "${YELLOW}[2/5]${NC} Updating dags_folder path..."
# Update dags_folder to point to our dags directory
sed -i "s|^dags_folder = .*|dags_folder = $PROJECT_DIR/dags|g" "$CONFIG_FILE"
echo "  ✓ dags_folder set to: $PROJECT_DIR/dags"
echo ""

echo -e "${YELLOW}[3/5]${NC} Disabling example DAGs..."
# Disable loading of example DAGs
sed -i 's/^load_examples = .*/load_examples = False/g' "$CONFIG_FILE"
echo "  ✓ load_examples set to False"
echo ""

echo -e "${YELLOW}[4/5]${NC} Installing missing dependencies..."
source venv/bin/activate
pip install virtualenv --quiet
echo "  ✓ virtualenv installed"
echo ""

echo -e "${YELLOW}[5/5]${NC} Verifying configuration..."
echo ""
echo "Current settings:"
echo "  dags_folder: $(grep '^dags_folder' $CONFIG_FILE | head -1)"
echo "  load_examples: $(grep '^load_examples' $CONFIG_FILE | head -1)"
echo ""

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}  Configuration Fixed!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Next steps:"
echo ""
echo "1. Stop both Airflow processes (Ctrl+C in both terminals)"
echo ""
echo "2. Clear any cached DAG files:"
echo "   rm -rf logs/*"
echo "   rm -f airflow.db-shm airflow.db-wal"
echo ""
echo "3. Restart Airflow:"
echo ""
echo "   Terminal 1:"
echo "   source venv/bin/activate"
echo "   export AIRFLOW_HOME=\$(pwd)"
echo "   airflow webserver --port 8080"
echo ""
echo "   Terminal 2:"
echo "   source venv/bin/activate"
echo "   export AIRFLOW_HOME=\$(pwd)"
echo "   airflow scheduler"
echo ""
echo -e "${GREEN}========================================${NC}"
