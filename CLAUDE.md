# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains a comprehensive Python library for measuring linguistic distances between languages using embedding spaces. The toolkit implements state-of-the-art methods for comparing monolingual corpora through word embeddings and geometric analysis, based on research from Mikolov et al. (2013) and recent advances in cross-lingual embedding alignment.

## Architecture and Components

### Core Library Structure (`src/`)

- **`data/`**: Data downloading and preprocessing
  - `downloader.py`: BibleDownloader for multilingual corpus collection
  - `preprocessor.py`: TextPreprocessor for cleaning and tokenizing text

- **`embeddings/`**: Embedding training and management
  - `trainer.py`: EmbeddingTrainer (Word2Vec, FastText via Gensim)
  - `loader.py`: EmbeddingLoader for loading and aligning embeddings

- **`alignment/`**: Embedding space alignment algorithms
  - `linear_mapping.py`: LinearMapping (linear regression, orthogonal, CCA)
  - `procrustes.py`: ProcrustesAlignment (orthogonal transformations)

- **`distance/`**: Distance measurement methods
  - `earth_movers.py`: EarthMoversDistance (Wasserstein distance)
  - `cosine_based.py`: CosineSimilarityMetrics (centroid, average pairwise)
  - `geometric.py`: GeometricDistances (Hausdorff, Chamfer, etc.)

- **`utils/`**: Utilities for visualization and I/O
  - `visualization.py`: Comprehensive plotting and chart creation
  - `io.py`: Data serialization and report generation

### Command Line Scripts (`scripts/`)

- `download_data.py`: Download and preprocess multilingual Bible data
- `train_embeddings.py`: Train Word2Vec/FastText embeddings
- `compute_distances.py`: Calculate linguistic distance matrices
- `visualize_results.py`: Create comprehensive visualizations

### Testing and Examples

- `test_pipeline.py`: Complete pipeline test with small datasets
- `examples/basic_usage.py`: Simple usage demonstration
- `examples/advanced_analysis.py`: Multi-method comparison example
- `tests/`: Unit tests for all components

## Common Development Commands

### Data Pipeline
```bash
# Download Bible translations for supported languages
python scripts/download_data.py --languages english spanish german italian dutch

# Train embeddings (Word2Vec example)
python scripts/train_embeddings.py --model-type word2vec --vector-size 150 --epochs 100

# Compute distance matrices
python scripts/compute_distances.py --distance-methods cosine_centroid hausdorff chamfer

# Create visualizations
python scripts/visualize_results.py --results-file results/linguistic_distances_*.json
```

### Testing
```bash
# Run complete pipeline test
python test_pipeline.py

# Run specific test modules  
python -m pytest tests/test_embeddings/
python -m pytest tests/test_distance/
python -m pytest tests/test_alignment/
```

### Package Management
```bash
# Install in development mode
pip install -e .

# Install with development dependencies
pip install -e ".[dev]"

# Build package
python -m build
```

## Key Technical Details

### Distance Metrics Implemented
1. **Cosine-based**: Centroid distances, average pairwise similarities
2. **Earth Mover's Distance**: Exact, Sinkhorn regularized, and approximation methods
3. **Geometric**: Hausdorff, Chamfer, Gromov-Hausdorff approximation
4. **Alignment-based**: MSE after Procrustes, linear, orthogonal alignment

### Embedding Methods
- Word2Vec (CBOW/Skip-gram) via Gensim
- FastText with character n-grams
- Configurable training parameters (vector size, window, epochs, etc.)

### Alignment Techniques
- Orthogonal Procrustes analysis
- Linear regression mapping (with regularization)
- Canonical Correlation Analysis (CCA)

### Supported Languages
Currently supports Bible translations in:
- English, Spanish, German, Italian, Dutch

## Data Flow

1. **Download**: `BibleDownloader` fetches multilingual Bible texts
2. **Preprocess**: `TextPreprocessor` cleans, tokenizes, filters text
3. **Train**: `EmbeddingTrainer` creates Word2Vec/FastText models
4. **Load**: `EmbeddingLoader` manages embeddings and vocabulary alignment
5. **Align**: Alignment classes transform embedding spaces
6. **Measure**: Distance classes compute linguistic distances
7. **Visualize**: Visualization utilities create plots and reports

## Code Patterns and Conventions

### Error Handling
- All major components include comprehensive error handling
- Methods return appropriate error values (inf, nan) for failed computations
- Logging is used throughout for debugging and progress tracking

### Performance Considerations
- Large embedding sets are sampled for expensive computations
- Distance matrices use symmetry to avoid redundant calculations
- Caching is implemented for loaded embeddings

### Configuration
- Training parameters configurable via command line or config files
- Modular design allows easy addition of new distance metrics
- Support for both dictionary and numpy array embedding formats

## Extension Points

### Adding New Languages
1. Add language-specific preprocessing rules in `TextPreprocessor`
2. Update `BibleDownloader` with new data sources
3. Add test cases and validation

### Adding New Distance Metrics
1. Create new class in appropriate `distance/` module
2. Implement `compute_distance_matrix()` method following existing patterns
3. Add CLI integration in `compute_distances.py`
4. Add visualization support if needed

### Adding New Alignment Methods
1. Implement in `alignment/` module with `fit_transform()` interface
2. Add evaluation methods for alignment quality
3. Integrate with distance computation pipeline

## Dependencies

### Core Requirements
- numpy, scipy, scikit-learn: Scientific computing
- gensim: Word2Vec/FastText training
- pandas: Data manipulation
- matplotlib, seaborn: Visualization
- requests, beautifulsoup4: Web scraping
- tqdm: Progress bars

### Optional Dependencies
- pot: Optimal transport (Earth Mover's Distance)
- pytest: Testing framework
- black, flake8: Code formatting and linting

## Current Implementation Status

✅ **Completed**: Full implementation of all core components
- Data downloading and preprocessing pipeline
- Word2Vec and FastText embedding training
- Multiple distance measurement algorithms
- Embedding space alignment methods
- Comprehensive visualization and reporting
- Command-line interface scripts
- Test pipeline and examples

The repository is production-ready with comprehensive documentation and examples.