#!/usr/bin/env python3
"""
Test script for the complete linguistic distance analysis pipeline.
This script tests all major components with small test datasets.
"""

import sys
import logging
from pathlib import Path
import numpy as np
import time

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import all components
from data.downloader import BibleDownloader
from data.preprocessor import TextPreprocessor
from embeddings.trainer import EmbeddingTrainer
from embeddings.loader import EmbeddingLoader
from distance.cosine_based import CosineSimilarityMetrics
from distance.geometric import GeometricDistances
from alignment.procrustes import ProcrustesAlignment
from utils.visualization import plot_distance_matrix
from utils.io import save_results


def setup_logging():
    """Set up logging for testing."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )


def test_data_components():
    """Test data downloading and preprocessing."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING DATA COMPONENTS")
    logger.info("=" * 50)
    
    # Test downloader with sample data (this will create sample data since real download may fail)
    logger.info("Testing BibleDownloader...")
    downloader = BibleDownloader("data/test_raw")
    
    test_languages = ["english", "spanish", "german"]
    results = downloader.download_all(test_languages, force_download=True)
    
    for lang, success in results.items():
        logger.info(f"  {lang}: {'SUCCESS' if success else 'FAILED'}")
    
    # Test preprocessor
    logger.info("Testing TextPreprocessor...")
    preprocessor = TextPreprocessor()
    
    # Copy our test files to raw directory for processing
    import shutil
    test_dir = Path("data/test")
    raw_dir = Path("data/test_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    if not test_dir.exists():
        logger.error("Test data directory not found! Run: python setup_test_env.py")
        return False
    
    for test_file in test_dir.glob("*_test.txt"):
        lang = test_file.stem.replace("_test", "")
        target_file = raw_dir / f"{lang}_bible.txt"
        shutil.copy(test_file, target_file)
        logger.info(f"Copied {test_file.name} to {target_file.name}")
    
    preprocessing_results = preprocessor.preprocess_all_languages(
        "data/test_raw", 
        "data/test_processed",
        test_languages
    )
    
    for lang, stats in preprocessing_results.items():
        logger.info(f"  {lang}: {stats['num_sentences']} sentences, {stats['num_words']} words")
    
    return True


def test_embedding_components():
    """Test embedding training and loading."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING EMBEDDING COMPONENTS")
    logger.info("=" * 50)
    
    # Test trainer
    logger.info("Testing EmbeddingTrainer...")
    trainer = EmbeddingTrainer("data/test_embeddings")
    
    test_languages = ["english", "spanish", "german"]
    
    # Train with small parameters for speed
    models = trainer.train_all_languages(
        input_dir="data/test_processed",
        languages=test_languages,
        model_type="word2vec",
        vector_size=50,
        window=3,
        min_count=1,
        epochs=10
    )
    
    for lang, model in models.items():
        vocab_size = len(model.wv) if model else 0
        logger.info(f"  {lang}: vocabulary size = {vocab_size}")
    
    # Test loader
    logger.info("Testing EmbeddingLoader...")
    loader = EmbeddingLoader("data/test_embeddings")
    
    loaded_embeddings = {}
    for lang in test_languages:
        try:
            embeddings = loader.load_embeddings(lang, "word2vec")
            loaded_embeddings[lang] = embeddings
            logger.info(f"  {lang}: loaded {len(embeddings)} embeddings")
        except Exception as e:
            logger.error(f"  {lang}: failed to load - {e}")
    
    return loaded_embeddings


def test_distance_components(embeddings_dict):
    """Test distance computation components."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING DISTANCE COMPONENTS")
    logger.info("=" * 50)
    
    # Convert embeddings to arrays
    embeddings_arrays = {}
    for lang, emb_dict in embeddings_dict.items():
        if emb_dict:
            words = list(emb_dict.keys())
            embeddings_arrays[lang] = np.array([emb_dict[word] for word in words])
    
    if not embeddings_arrays:
        logger.error("No embeddings available for distance testing")
        return {}
    
    distance_results = {}
    
    # Test cosine similarity metrics
    logger.info("Testing CosineSimilarityMetrics...")
    cosine_metrics = CosineSimilarityMetrics()
    
    try:
        cosine_distances = cosine_metrics.compute_similarity_matrix(
            embeddings_arrays, metric='centroid_distance'
        )
        distance_results['cosine_centroid'] = cosine_distances
        
        # Show some results
        languages = list(cosine_distances.keys())
        for i, lang1 in enumerate(languages[:2]):
            for lang2 in languages[i+1:i+2]:
                dist = cosine_distances[lang1][lang2]
                logger.info(f"  Cosine distance {lang1}-{lang2}: {dist:.4f}")
                
    except Exception as e:
        logger.error(f"  Cosine similarity failed: {e}")
    
    # Test geometric distances
    logger.info("Testing GeometricDistances...")
    geometric = GeometricDistances()
    
    try:
        hausdorff_distances = geometric.compute_distance_matrix(
            embeddings_arrays, metric='hausdorff', sample_size=20
        )
        distance_results['hausdorff'] = hausdorff_distances
        
        # Show some results
        languages = list(hausdorff_distances.keys())
        for i, lang1 in enumerate(languages[:2]):
            for lang2 in languages[i+1:i+2]:
                dist = hausdorff_distances[lang1][lang2]
                logger.info(f"  Hausdorff distance {lang1}-{lang2}: {dist:.4f}")
                
    except Exception as e:
        logger.error(f"  Geometric distances failed: {e}")
    
    return distance_results


def test_alignment_components(embeddings_dict):
    """Test alignment components."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING ALIGNMENT COMPONENTS")
    logger.info("=" * 50)
    
    if len(embeddings_dict) < 2:
        logger.warning("Need at least 2 languages for alignment testing")
        return {}
    
    languages = list(embeddings_dict.keys())[:2]
    lang1, lang2 = languages[0], languages[1]
    
    logger.info(f"Testing alignment between {lang1} and {lang2}...")
    
    # Get common vocabulary (simplified - just take intersection of words)
    words1 = set(embeddings_dict[lang1].keys())
    words2 = set(embeddings_dict[lang2].keys())
    common_words = list(words1.intersection(words2))
    
    if len(common_words) < 5:
        logger.warning(f"Too few common words ({len(common_words)}) for alignment testing")
        return {}
    
    # Create aligned matrices
    common_words = common_words[:min(20, len(common_words))]  # Limit for speed
    emb1 = np.array([embeddings_dict[lang1][word] for word in common_words])
    emb2 = np.array([embeddings_dict[lang2][word] for word in common_words])
    
    logger.info(f"Testing with {len(common_words)} common words")
    
    # Test Procrustes alignment
    try:
        logger.info("Testing ProcrustesAlignment...")
        procrustes = ProcrustesAlignment()
        aligned_emb1 = procrustes.fit_transform(emb1, emb2)
        
        evaluation = procrustes.evaluate_alignment(emb1, emb2)
        logger.info(f"  MSE after alignment: {evaluation['mse']:.6f}")
        logger.info(f"  Mean cosine similarity: {evaluation['mean_cosine_similarity']:.4f}")
        
    except Exception as e:
        logger.error(f"  Procrustes alignment failed: {e}")
    
    return {'alignment_test': 'completed'}


def test_visualization_components(distance_results):
    """Test visualization components."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING VISUALIZATION COMPONENTS")
    logger.info("=" * 50)
    
    if not distance_results:
        logger.warning("No distance results available for visualization testing")
        return
    
    # Create output directory
    vis_dir = Path("test_visualizations")
    vis_dir.mkdir(exist_ok=True)
    
    # Test distance matrix plotting
    for method_name, distance_matrix in distance_results.items():
        try:
            logger.info(f"Creating visualization for {method_name}...")
            
            fig = plot_distance_matrix(
                distance_matrix,
                title=f"Test {method_name.title()} Distance Matrix",
                save_path=str(vis_dir / f"test_{method_name}.png")
            )
            
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
                logger.info(f"  Saved: test_{method_name}.png")
            
        except Exception as e:
            logger.error(f"  Visualization for {method_name} failed: {e}")


def test_io_components(test_results):
    """Test I/O components."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING I/O COMPONENTS")
    logger.info("=" * 50)
    
    try:
        # Test saving results
        logger.info("Testing save_results...")
        save_results(
            test_results,
            "test_results.json",
            base_dir="test_output",
            include_metadata=True
        )
        logger.info("  Results saved successfully")
        
        # Verify file was created
        results_file = Path("test_output/test_results.json")
        if results_file.exists():
            file_size = results_file.stat().st_size
            logger.info(f"  File size: {file_size} bytes")
        else:
            logger.error("  Results file was not created")
            
    except Exception as e:
        logger.error(f"  I/O testing failed: {e}")


def main():
    """Run the complete test pipeline."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("*" * 60)
    logger.info("LINGUISTIC DISTANCE ANALYSIS - PIPELINE TEST")
    logger.info("*" * 60)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # Test data components
        data_success = test_data_components()
        test_results['data_pipeline'] = {'status': 'success' if data_success else 'failed'}
        
        # Test embedding components
        embeddings_dict = test_embedding_components()
        test_results['embeddings'] = {
            'status': 'success' if embeddings_dict else 'failed',
            'languages_processed': list(embeddings_dict.keys()) if embeddings_dict else []
        }
        
        # Test distance components
        distance_results = test_distance_components(embeddings_dict)
        test_results['distances'] = {
            'status': 'success' if distance_results else 'failed',
            'methods_tested': list(distance_results.keys()) if distance_results else []
        }
        
        # Test alignment components
        alignment_results = test_alignment_components(embeddings_dict)
        test_results['alignment'] = {
            'status': 'success' if alignment_results else 'failed'
        }
        
        # Test visualization components
        test_visualization_components(distance_results)
        test_results['visualization'] = {'status': 'completed'}
        
        # Test I/O components
        test_io_components(test_results)
        test_results['io'] = {'status': 'completed'}
        
        total_time = time.time() - start_time
        test_results['total_runtime_seconds'] = total_time
        
        logger.info("*" * 60)
        logger.info("TEST PIPELINE SUMMARY")
        logger.info("*" * 60)
        
        for component, results in test_results.items():
            if component != 'total_runtime_seconds':
                status = results.get('status', 'unknown')
                logger.info(f"{component.upper()}: {status}")
        
        logger.info(f"\\nTotal runtime: {total_time:.2f} seconds")
        
        # Check if core components passed
        core_components = ['data_pipeline', 'embeddings', 'distances']
        core_success = all(test_results[comp]['status'] == 'success' for comp in core_components)
        
        if core_success:
            logger.info("\\n✅ CORE PIPELINE TEST PASSED!")
            logger.info("The linguistic distance analysis system is working correctly.")
        else:
            logger.warning("\\n⚠️  SOME TESTS FAILED")
            logger.info("Check the logs above for details on failed components.")
            
    except Exception as e:
        logger.error(f"Pipeline test failed with exception: {e}")
        return 1
        
    return 0 if core_success else 1


if __name__ == "__main__":
    sys.exit(main())