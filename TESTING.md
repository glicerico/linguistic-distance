# Testing the Linguistic Distance Repository

This guide explains how to test the linguistic distance measurement implementation.

## Prerequisites

The test pipeline requires Python scientific packages. You **must** set up a virtual environment first.

## Quick Setup

### Option 1: Using Python venv

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# venv\Scripts\activate   # On Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up test data
python setup_test_env.py

# 4. Run tests
python test_pipeline.py
```

### Option 2: Using conda

```bash
# 1. Create and activate conda environment
conda create -n linguistic_distance python=3.8 -y
conda activate linguistic_distance

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up test data
python setup_test_env.py

# 4. Run tests
python test_pipeline.py
```

## Test Components

### Structure Test (No dependencies required)
```bash
# Test file structure and Python syntax
python basic_structure_test.py
```

### Minimal Test (Limited dependencies)
```bash
# Basic functionality test with error handling
python test_pipeline_minimal.py
```

### Full Test Pipeline (All dependencies required)
```bash
# Complete end-to-end pipeline test
python test_pipeline.py
```

## Test Data

The tests use small multilingual datasets located in `data/test/`:
- `english_test.txt`: 20 simple English sentences
- `spanish_test.txt`: 20 Spanish translations  
- `german_test.txt`: 20 German translations

These provide enough data to test:
- Text preprocessing
- Embedding training (with small parameters)
- Distance computation
- Visualization generation

## Expected Test Results

### With Virtual Environment + Dependencies

```
✅ CORE PIPELINE TEST PASSED!
The linguistic distance analysis system is working correctly.

Components tested:
- Data pipeline: SUCCESS
- Embeddings: SUCCESS (3 languages processed)  
- Distances: SUCCESS (3 methods tested)
- Alignment: SUCCESS
- Visualization: completed
- I/O: completed

Total runtime: ~30-60 seconds
```

### Without Dependencies

```
❌ ALL TESTS FAILED!
Check your setup and dependencies.

Missing: NumPy, SciPy, scikit-learn, Gensim, Matplotlib, Pandas, tqdm
```

## Troubleshooting

### Virtual Environment Issues

If you get import errors even after installing dependencies:

```bash
# Make sure virtual environment is activated
which python  # Should show path to venv/bin/python

# Reinstall dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Check installed packages
pip list
```

### Memory Issues

If embedding training fails due to memory:

```bash
# The test uses very small parameters:
# - vector_size=50 (instead of 150)
# - epochs=10 (instead of 100) 
# - min_count=1 (instead of 5)

# This should work on systems with 2GB+ RAM
```

### Missing Test Data

If you see "Test data directory not found":

```bash
# Run the setup script
python setup_test_env.py

# Manually check test files exist
ls data/test/*.txt
```

## Test Details

### What Each Test Does

1. **Data Components**
   - Downloads/creates sample Bible texts
   - Preprocesses text (tokenization, cleaning)
   - Validates output statistics

2. **Embedding Components**  
   - Trains Word2Vec embeddings (small parameters)
   - Tests vocabulary extraction
   - Validates model loading/saving

3. **Distance Components**
   - Computes cosine similarity matrices
   - Tests Hausdorff distances  
   - Validates distance matrix symmetry

4. **Alignment Components**
   - Tests Procrustes alignment between language pairs
   - Validates alignment quality metrics
   - Tests common vocabulary extraction

5. **Visualization Components**
   - Creates distance matrix heatmaps
   - Tests plot generation and saving
   - Validates output file creation

6. **I/O Components**
   - Tests result serialization (JSON)
   - Validates file I/O operations
   - Tests metadata inclusion

## Performance Expectations

- **Test data size**: ~1KB per language (very small)
- **Embedding training**: ~5-10 seconds per language  
- **Distance computation**: ~1-2 seconds
- **Total runtime**: 30-60 seconds
- **Memory usage**: <500MB

The tests are designed to be fast while validating all core functionality.

## Next Steps After Testing

Once tests pass:

1. **Try examples**:
   ```bash
   python examples/basic_usage.py
   python examples/advanced_analysis.py
   ```

2. **Download real data**:
   ```bash
   python scripts/download_data.py
   ```

3. **Train full models**:
   ```bash
   python scripts/train_embeddings.py --vector-size 150 --epochs 100
   ```

4. **Analyze results**:
   ```bash
   python scripts/compute_distances.py
   python scripts/visualize_results.py --results-file results/linguistic_distances_*.json
   ```