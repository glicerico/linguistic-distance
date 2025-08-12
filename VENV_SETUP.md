
# VIRTUAL ENVIRONMENT SETUP INSTRUCTIONS

## Option 1: Using venv (Python built-in)
```bash
# Create virtual environment
python3 -m venv linguistic_distance_venv

# Activate virtual environment
# On Linux/Mac:
source linguistic_distance_venv/bin/activate
# On Windows:
# linguistic_distance_venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_pipeline.py
```

## Option 2: Using conda
```bash
# Create conda environment
conda create -n linguistic_distance python=3.8

# Activate environment
conda activate linguistic_distance

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_pipeline.py
```

## Quick Test (after activation)
```bash
# Test basic imports
python -c "import numpy, scipy, gensim; print('Core packages available!')"

# Run pipeline test
python test_pipeline.py
```
