# Linguistic Distance Measurement

A comprehensive Python library for measuring linguistic distances between languages using embedding spaces. This toolkit implements state-of-the-art methods for comparing monolingual corpora through word embeddings and geometric analysis.

## 📖 Overview

This library provides tools to:
- **Download and preprocess** multilingual corpora (Bible translations)
- **Train word embeddings** (Word2Vec, FastText) on monolingual data
- **Align embedding spaces** using various mathematical techniques  
- **Compute linguistic distances** using multiple metrics
- **Visualize results** with comprehensive plots and analysis

The methods are based on research from [Mikolov et al. (2013)](https://arxiv.org/abs/1309.4168) and recent advances in cross-lingual embedding alignment.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/linguistic-distance.git
cd linguistic-distance

# Install dependencies
pip install -r requirements.txt

# Optional: Install in development mode (allows editing source code without reinstalling)
# This creates a live link to the source code for active development
pip install -e .
```

### Basic Usage

```python
from data.downloader import BibleDownloader
from embeddings.trainer import EmbeddingTrainer
from distance.cosine_based import CosineSimilarityMetrics
from utils.visualization import plot_distance_matrix

# Download and preprocess data
downloader = BibleDownloader()
downloader.download_all(["english", "spanish", "german"])

# Train embeddings
trainer = EmbeddingTrainer()
trainer.train_all_languages(model_type="word2vec")

# Compute distances
cosine_metrics = CosineSimilarityMetrics()
distances = cosine_metrics.compute_similarity_matrix(embeddings_dict)

# Visualize results
plot_distance_matrix(distances, save_path="distance_matrix.png")
```

### Command Line Interface

```bash
# Download data for all supported languages
python scripts/download_data.py

# Train Word2Vec embeddings
python scripts/train_embeddings.py --model-type word2vec --vector-size 150

# Compute linguistic distances
python scripts/compute_distances.py --distance-methods cosine_centroid hausdorff

# Create visualizations
python scripts/visualize_results.py --results-file results/linguistic_distances.json
```

## 📊 Supported Languages

Currently supports Bible translations in:
- **English** 🇺🇸
- **Spanish** 🇪🇸  
- **German** 🇩🇪
- **Italian** 🇮🇹
- **Dutch** 🇳🇱

## 🔬 Distance Metrics

### Direct Distance Methods
- **Cosine Similarity**: Centroid-based and average pairwise similarities
- **Earth Mover's Distance**: Wasserstein distance between embedding distributions  
- **Hausdorff Distance**: Maximum distance between embedding point clouds
- **Chamfer Distance**: Average nearest-neighbor distances
- **Geometric Methods**: Gromov-Hausdorff approximations, shape context

### Alignment-Based Methods
- **Procrustes Analysis**: Optimal orthogonal transformations
- **Linear Mapping**: Least-squares linear transformations
- **Canonical Correlation Analysis**: Statistical alignment techniques

## 🧮 Technical Details

### Embedding Methods
- **Word2Vec**: CBOW and Skip-gram algorithms via Gensim
- **FastText**: Subword-aware embeddings with character n-grams

### Alignment Techniques
- Orthogonal Procrustes for isometric transformations
- Regularized linear regression for flexible mappings
- Cross-validation and bootstrap confidence intervals

### Distance Computation
- Efficient algorithms with sampling for large vocabularies
- Parallel processing for multiple language pairs
- Robust handling of missing vocabulary and edge cases

## 📁 Repository Structure

```
linguistic-distance/
├── src/                          # Core library code
│   ├── data/                     # Data downloading and preprocessing
│   ├── embeddings/               # Embedding training and loading
│   ├── alignment/                # Space alignment algorithms  
│   ├── distance/                 # Distance measurement methods
│   └── utils/                    # Visualization and I/O utilities
├── scripts/                      # Command-line interface scripts
├── examples/                     # Usage examples and tutorials
├── tests/                        # Test suite
├── data/                         # Data storage directories
│   ├── raw/                      # Downloaded raw texts
│   ├── processed/                # Preprocessed training data
│   ├── embeddings/               # Trained embedding models
│   └── test/                     # Small test datasets
└── results/                      # Analysis outputs and reports
```

## 📈 Example Results

### Distance Matrix Visualization
![Distance Matrix Example](https://via.placeholder.com/400x300?text=Distance+Matrix+Heatmap)

### Embedding Space Visualization  
![Embedding Visualization](https://via.placeholder.com/400x300?text=2D+Embedding+Projection)

### Linguistic Dendrogram
![Dendrogram Example](https://via.placeholder.com/400x300?text=Hierarchical+Clustering)

## 🔧 Configuration

### Environment Variables
```bash
export LINGUISTIC_DISTANCE_DATA_DIR="/path/to/data"
export LINGUISTIC_DISTANCE_RESULTS_DIR="/path/to/results"
```

### Training Parameters
```python
# Word2Vec configuration
{
    "vector_size": 150,
    "window": 5,
    "min_count": 5,
    "epochs": 100,
    "sg": 0  # CBOW
}

# FastText configuration  
{
    "vector_size": 150,
    "window": 5,
    "min_count": 5,
    "epochs": 100,
    "sg": 0,
    "min_n": 3,
    "max_n": 6
}
```

## 🧪 Testing

**⚠️ Important: Tests require a virtual environment with dependencies installed.**

### Quick Test Setup

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 2. Install dependencies  
pip install -r requirements.txt

# 3. Set up test data
python setup_test_env.py

# 4. Run tests
python test_pipeline.py
```

### Test Options

```bash
# Verify structure and test data (no dependencies needed)
python verify_test_data.py

# Minimal test with dependency checks
python test_pipeline_minimal.py

# Complete pipeline test (requires all dependencies)
python test_pipeline.py
```

See [TESTING.md](TESTING.md) for detailed testing instructions.

## 📚 Examples

### Basic Analysis
```bash
python examples/basic_usage.py
```

### Advanced Multi-Method Comparison
```bash
python examples/advanced_analysis.py
```

### Custom Analysis
```python
from embeddings.loader import EmbeddingLoader
from distance.earth_movers import EarthMoversDistance

loader = EmbeddingLoader()
embeddings = {
    lang: loader.load_embeddings(lang, "word2vec") 
    for lang in ["english", "spanish"]
}

emd = EarthMoversDistance(method="sinkhorn")
distance = emd.compute_emd(embeddings["english"], embeddings["spanish"])
```

## 🔬 Research Background

This library implements methods from several key papers:

1. **Mikolov, T., Le, Q. V., & Sutskever, I. (2013)**. [Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168). *arXiv preprint arXiv:1309.4168*.

2. **Schönemann, P. H. (1966)**. A generalized solution of the orthogonal procrustes problem. *Psychometrika, 31*(1), 1-10. - Foundational work on Procrustes analysis for embedding alignment.

3. **Rubner, Y., Tomasi, C., & Guibas, L. J. (2000)**. The earth mover's distance as a metric for image retrieval. *International journal of computer vision, 40*(2), 99-121. - Earth Mover's Distance for distribution comparison.

4. **Cuturi, M. (2013)**. Sinkhorn distances: Lightspeed computation of optimal transport. *Advances in neural information processing systems, 26*. - Efficient computation of Wasserstein distances.

5. **Conneau, A., Lample, G., Ranzato, M. A., Denoyer, L., & Jégou, H. (2017)**. [Word translation without parallel data](https://arxiv.org/abs/1710.04087). *arXiv preprint arXiv:1710.04087*. - Modern cross-lingual embedding alignment techniques.

6. **Grave, E., Joulin, A., & Berthet, Q. (2018)**. [Unsupervised alignment of embeddings with wasserstein procrustes](https://arxiv.org/abs/1805.11222). *arXiv preprint arXiv:1805.11222*. - Combining optimal transport with Procrustes alignment.

### Citation

If you use this library in your research, please cite:

```bibtex
@software{linguistic_distance_2024,
  title={Linguistic Distance Measurement Library},
  author={Your Name},
  year={2024},
  url={https://github.com/your-username/linguistic-distance}
}
```

## 🤝 Contributing

Contributions are welcome! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/your-username/linguistic-distance.git
cd linguistic-distance

# Install in development mode with development dependencies
# Development mode creates a live link to source code for active development
pip install -e ".[dev]"
pre-commit install
```

### Adding New Languages
1. Add language-specific preprocessing rules in `src/data/preprocessor.py`
2. Update `BibleDownloader` with new data sources
3. Add test cases for the new language

### Adding New Distance Metrics
1. Implement the metric in the appropriate module under `src/distance/`
2. Add unit tests in `tests/test_distance/`
3. Update the CLI scripts to support the new metric

## 🐛 Issues and Support

- **Bug Reports**: [GitHub Issues](https://github.com/your-username/linguistic-distance/issues)
- **Feature Requests**: [GitHub Discussions](https://github.com/your-username/linguistic-distance/discussions)
- **Documentation**: [Wiki](https://github.com/your-username/linguistic-distance/wiki)

## 🙏 Acknowledgments

- Bible text data from various open-source repositories
- Gensim library for embedding training
- Scientific Python ecosystem (NumPy, SciPy, scikit-learn)
- Visualization tools (Matplotlib, Seaborn)

---

**Keywords**: Computational Linguistics, Cross-lingual Embeddings, Distance Metrics, Natural Language Processing, Word2Vec, FastText, Embedding Alignment
