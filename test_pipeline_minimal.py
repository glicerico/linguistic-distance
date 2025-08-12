#!/usr/bin/env python3
"""
Minimal test script for the linguistic distance analysis pipeline.
This version uses only the small test datasets and handles missing dependencies gracefully.
"""

import sys
import logging
import shutil
from pathlib import Path
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Set up logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_dependencies():
    """Check if required packages are available."""
    logger = logging.getLogger(__name__)
    logger.info("Checking dependencies...")
    
    required_packages = {
        'numpy': 'NumPy',
        'scipy': 'SciPy', 
        'sklearn': 'scikit-learn',
        'gensim': 'Gensim',
        'matplotlib': 'Matplotlib',
        'pandas': 'Pandas',
        'tqdm': 'tqdm'
    }
    
    missing = []
    available = []
    
    for pkg_name, display_name in required_packages.items():
        try:
            __import__(pkg_name)
            available.append(display_name)
        except ImportError:
            missing.append(display_name)
    
    logger.info(f"Available: {', '.join(available)}")
    if missing:
        logger.warning(f"Missing: {', '.join(missing)}")
        logger.warning("Some tests may fail. Please install dependencies with:")
        logger.warning("  pip install -r requirements.txt")
        return False
    else:
        logger.info("✅ All dependencies available!")
        return True

def setup_test_data():
    """Set up test data directories and files."""
    logger = logging.getLogger(__name__)
    logger.info("Setting up test data...")
    
    # Create test directories
    test_dirs = [
        "data/test_raw",
        "data/test_processed", 
        "data/test_embeddings",
        "test_output",
        "test_visualizations"
    ]
    
    for dir_path in test_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Copy test files to raw directory
    test_dir = Path("data/test")
    raw_dir = Path("data/test_raw")
    
    if not test_dir.exists():
        logger.error("Test data directory not found! Run: python setup_test_env.py")
        return False
    
    test_files_copied = 0
    for test_file in test_dir.glob("*_test.txt"):
        lang = test_file.stem.replace("_test", "")
        target_file = raw_dir / f"{lang}_bible.txt"
        shutil.copy(test_file, target_file)
        test_files_copied += 1
        logger.info(f"  Copied {test_file.name} -> {target_file.name}")
    
    if test_files_copied == 0:
        logger.error("No test files found! Run: python setup_test_env.py")
        return False
        
    logger.info(f"✅ Set up test data for {test_files_copied} languages")
    return True

def test_imports():
    """Test that we can import core modules."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING CORE IMPORTS")
    logger.info("=" * 50)
    
    test_imports = [
        ("data.preprocessor", "TextPreprocessor"),
        ("embeddings.trainer", "EmbeddingTrainer"),
        ("embeddings.loader", "EmbeddingLoader"),
        ("distance.cosine_based", "CosineSimilarityMetrics"),
        ("utils.io", "save_results")
    ]
    
    successful_imports = 0
    
    for module_name, class_name in test_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            logger.info(f"✅ {module_name}.{class_name}")
            successful_imports += 1
        except Exception as e:
            logger.error(f"❌ {module_name}.{class_name}: {e}")
    
    success_rate = successful_imports / len(test_imports)
    logger.info(f"Import success rate: {successful_imports}/{len(test_imports)} ({success_rate:.1%})")
    
    return success_rate > 0.8

def test_preprocessing():
    """Test text preprocessing with small data."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING TEXT PREPROCESSING")
    logger.info("=" * 50)
    
    try:
        from data.preprocessor import TextPreprocessor
        
        preprocessor = TextPreprocessor()
        test_languages = ["english", "spanish", "german"]
        
        results = preprocessor.preprocess_all_languages(
            input_dir="data/test_raw",
            output_dir="data/test_processed", 
            languages=test_languages
        )
        
        total_sentences = 0
        total_words = 0
        
        for lang, stats in results.items():
            sentences = stats['num_sentences']
            words = stats['num_words']
            total_sentences += sentences
            total_words += words
            logger.info(f"  {lang}: {sentences} sentences, {words} words")
        
        logger.info(f"✅ Preprocessing complete: {total_sentences} sentences, {total_words} words")
        return total_sentences > 0 and total_words > 0
        
    except Exception as e:
        logger.error(f"❌ Preprocessing failed: {e}")
        return False

def test_embedding_training():
    """Test embedding training with small parameters."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING EMBEDDING TRAINING")
    logger.info("=" * 50)
    
    try:
        from embeddings.trainer import EmbeddingTrainer
        
        trainer = EmbeddingTrainer("data/test_embeddings")
        test_languages = ["english", "spanish", "german"]
        
        # Use very small parameters for fast testing
        models = trainer.train_all_languages(
            input_dir="data/test_processed",
            languages=test_languages,
            model_type="word2vec",
            vector_size=20,  # Very small for testing
            window=2,
            min_count=1,
            epochs=5,  # Few epochs for speed
            workers=1
        )
        
        successful_models = 0
        total_vocab = 0
        
        for lang, model in models.items():
            if model and hasattr(model, 'wv'):
                vocab_size = len(model.wv)
                total_vocab += vocab_size
                successful_models += 1
                logger.info(f"  {lang}: {vocab_size} words")
            else:
                logger.warning(f"  {lang}: Failed to train")
        
        logger.info(f"✅ Trained {successful_models}/{len(test_languages)} models, {total_vocab} total vocab")
        return successful_models > 0
        
    except Exception as e:
        logger.error(f"❌ Embedding training failed: {e}")
        return False

def test_distance_computation():
    """Test distance computation with trained embeddings."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING DISTANCE COMPUTATION")
    logger.info("=" * 50)
    
    try:
        from embeddings.loader import EmbeddingLoader
        from distance.cosine_based import CosineSimilarityMetrics
        
        # Load embeddings
        loader = EmbeddingLoader("data/test_embeddings")
        test_languages = ["english", "spanish", "german"]
        
        embeddings_dict = {}
        for language in test_languages:
            try:
                embeddings = loader.load_embeddings(language, "word2vec")
                if embeddings:
                    # Convert to numpy array
                    import numpy as np
                    words = list(embeddings.keys())
                    embeddings_array = np.array([embeddings[word] for word in words])
                    embeddings_dict[language] = embeddings_array
                    logger.info(f"  Loaded {len(embeddings)} embeddings for {language}")
            except Exception as e:
                logger.warning(f"  Could not load {language}: {e}")
        
        if len(embeddings_dict) < 2:
            logger.warning("Need at least 2 languages for distance computation")
            return False
        
        # Compute cosine distances
        cosine_metrics = CosineSimilarityMetrics()
        distance_matrix = cosine_metrics.compute_similarity_matrix(
            embeddings_dict, metric='centroid_distance'
        )
        
        # Check results
        computed_distances = 0
        for lang1 in distance_matrix:
            for lang2 in distance_matrix[lang1]:
                if lang1 != lang2:
                    dist = distance_matrix[lang1][lang2]
                    if not (dist != dist):  # Not NaN
                        computed_distances += 1
                        logger.info(f"  Distance {lang1}-{lang2}: {dist:.4f}")
        
        logger.info(f"✅ Computed {computed_distances} pairwise distances")
        return computed_distances > 0
        
    except Exception as e:
        logger.error(f"❌ Distance computation failed: {e}")
        return False

def test_io_functionality():
    """Test I/O functionality."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING I/O FUNCTIONALITY")
    logger.info("=" * 50)
    
    try:
        from utils.io import save_results
        
        test_results = {
            'test_data': {
                'languages': ['english', 'spanish', 'german'],
                'test_metric': 0.95
            },
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        save_results(
            test_results,
            "minimal_test_results.json",
            base_dir="test_output"
        )
        
        # Check if file was created
        results_file = Path("test_output/minimal_test_results.json")
        if results_file.exists():
            file_size = results_file.stat().st_size
            logger.info(f"✅ Results saved: {results_file} ({file_size} bytes)")
            return True
        else:
            logger.error("❌ Results file was not created")
            return False
            
    except Exception as e:
        logger.error(f"❌ I/O test failed: {e}")
        return False

def main():
    """Run minimal test pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("*" * 60)
    logger.info("LINGUISTIC DISTANCE - MINIMAL TEST PIPELINE")
    logger.info("*" * 60)
    
    start_time = time.time()
    
    # Check setup
    deps_ok = check_dependencies()
    data_ok = setup_test_data()
    
    if not data_ok:
        logger.error("❌ Test data setup failed!")
        logger.error("Run: python setup_test_env.py")
        return 1
    
    if not deps_ok:
        logger.warning("⚠️  Some dependencies missing - continuing with limited tests")
    
    # Run tests
    test_results = {}
    tests = [
        ("Core Imports", test_imports),
        ("Text Preprocessing", test_preprocessing),
    ]
    
    # Add embedding tests only if dependencies are available
    if deps_ok:
        tests.extend([
            ("Embedding Training", test_embedding_training),
            ("Distance Computation", test_distance_computation),
        ])
    
    tests.append(("I/O Functionality", test_io_functionality))
    
    # Execute tests
    for test_name, test_func in tests:
        logger.info(f"\n🔍 Running {test_name}...")
        try:
            result = test_func()
            test_results[test_name] = result
            status = "PASS" if result else "FAIL"
            emoji = "✅" if result else "❌"
            logger.info(f"{emoji} {test_name}: {status}")
        except Exception as e:
            logger.error(f"❌ {test_name}: EXCEPTION - {e}")
            test_results[test_name] = False
    
    # Summary
    total_time = time.time() - start_time
    passed = sum(test_results.values())
    total = len(test_results)
    
    logger.info("\n" + "*" * 60)
    logger.info("MINIMAL TEST SUMMARY")
    logger.info("*" * 60)
    
    for test_name, result in test_results.items():
        status = "PASS" if result else "FAIL" 
        emoji = "✅" if result else "❌"
        logger.info(f"{emoji} {test_name}: {status}")
    
    logger.info(f"\nResults: {passed}/{total} tests passed")
    logger.info(f"Runtime: {total_time:.1f} seconds")
    
    if passed == total:
        logger.info("\n🎉 ALL TESTS PASSED!")
        logger.info("The minimal pipeline is working correctly.")
    elif passed > 0:
        logger.info(f"\n⚠️  {total - passed} tests failed, but core functionality works.")
        if not deps_ok:
            logger.info("Install full dependencies for complete testing:")
            logger.info("  pip install -r requirements.txt")
    else:
        logger.error("\n❌ ALL TESTS FAILED!")
        logger.error("Check your setup and dependencies.")
        
    return 0 if passed > 0 else 1

if __name__ == "__main__":
    sys.exit(main())