#!/bin/bash

# Multi-Model Forecasting - Requirements Installation Script for WSL/Linux
# This script installs all dependencies for Prophet, SARIMA, DeepAR, and Ensemble models

# Exit immediately if a command exits with a non-zero status
# set -e 

# Colors for output
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  MULTI-MODEL FORECASTING - DEPENDENCY INSTALLATION (WSL/Linux)${NC}"
echo -e "${CYAN}===============================================================================${NC}"
echo ""

# Ensure we are in the script's directory
cd "$(dirname "$0")"
echo -e "${YELLOW}Current Directory: $(pwd)${NC}"
ls -F requirements/

# Check python version
echo -e "${YELLOW}Using Python:${NC}"
which python
python --version

# Update pip
echo -e "${CYAN}[INFO] Updating pip...${NC}"
python -m pip install --upgrade pip

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 1: Installing Base Requirements${NC}"
echo -e "${CYAN}===============================================================================${NC}"
python -m pip install -r requirements/base.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Base requirements installed!${NC}"
else
    echo -e "${RED}[ERROR] Failed to install base requirements!${NC}"
    exit 1
fi

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 2: Installing Prophet Requirements${NC}"
echo -e "${CYAN}===============================================================================${NC}"
python -m pip install -r requirements/prophet.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Prophet requirements installed!${NC}"
else
    echo -e "${YELLOW}[WARNING] Failed to install Prophet requirements.${NC}"
fi

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 3: Installing SARIMA Requirements${NC}"
echo -e "${CYAN}===============================================================================${NC}"
python -m pip install -r requirements/sarima.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] SARIMA requirements installed!${NC}"
else
    echo -e "${YELLOW}[WARNING] Failed to install SARIMA requirements.${NC}"
fi

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 4: Installing Ensemble Requirements${NC}"
echo -e "${CYAN}===============================================================================${NC}"
python -m pip install -r requirements/ensemble.txt
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] Ensemble requirements installed!${NC}"
else
    echo -e "${YELLOW}[WARNING] Failed to install Ensemble requirements.${NC}"
fi

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 5: Installing DeepAR Requirements (PyTorch + CUDA for RTX 4050)${NC}"
echo -e "${CYAN}===============================================================================${NC}"

echo -e "${YELLOW}[INFO] DeepAR requires PyTorch with CUDA support for GPU acceleration.${NC}"
echo -e "${CYAN}[INFO] Installing PyTorch with CUDA 11.8 support...${NC}"

# Install PyTorch specifically for Linux/WSL with CUDA 11.8
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if [ $? -eq 0 ]; then
    echo -e "${GREEN}[SUCCESS] PyTorch with CUDA installed!${NC}"
    
    echo -e "${CYAN}[INFO] Installing GluonTS and other DeepAR dependencies...${NC}"
    python -m pip install -r requirements/deepar.txt
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}[SUCCESS] DeepAR requirements installed!${NC}"
    else
        echo -e "${YELLOW}[WARNING] Failed to install DeepAR requirements.${NC}"
    fi
else
    echo -e "${RED}[ERROR] Failed to install PyTorch. DeepAR will not be available.${NC}"
fi

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  STEP 6: Verification${NC}"
echo -e "${CYAN}===============================================================================${NC}"

echo -e "${YELLOW}Python Version:${NC}"
python --version

echo ""
echo -e "${YELLOW}Verifying Key Packages:${NC}"
python -c "import pandas; print(f'✓ pandas {pandas.__version__}')" || echo "✗ pandas not installed"
python -c "import numpy; print(f'✓ numpy {numpy.__version__}')" || echo "✗ numpy not installed"
python -c "try: import prophet; print(f'✓ prophet installed'); except: print('✗ prophet not installed')"
python -c "try: import pmdarima; print(f'✓ pmdarima installed'); except: print('✗ pmdarima not installed')"
python -c "try: import torch; print(f'✓ torch {torch.__version__}'); print(f'  CUDA Available: {torch.cuda.is_available()}'); except: print('✗ torch not installed')"
python -c "try: import gluonts; print(f'✓ gluonts installed'); except: print('✗ gluonts not installed')"

echo ""
echo -e "${CYAN}===============================================================================${NC}"
echo -e "${GREEN}  INSTALLATION COMPLETE!${NC}"
echo -e "${CYAN}===============================================================================${NC}"
