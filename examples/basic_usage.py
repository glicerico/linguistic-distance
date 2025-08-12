#!/usr/bin/env python3
"""
Basic usage example for the linguistic distance analysis library.

This example demonstrates how to:
1. Download and preprocess multilingual data
2. Train word embeddings
3. Compute linguistic distances
4. Visualize results
"""

import sys
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import components
from data.downloader import BibleDownloader
from data.preprocessor import TextPreprocessor
from embeddings.trainer import EmbeddingTrainer
from embeddings.loader import EmbeddingLoader
from distance.cosine_based import CosineSimilarityMetrics
from utils.visualization import plot_distance_matrix
from utils.io import save_results

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Basic usage example."""
    logger.info("=== LINGUISTIC DISTANCE ANALYSIS - BASIC EXAMPLE ===")
    
    # Step 1: Download and preprocess data
    logger.info("Step 1: Downloading and preprocessing data...")
    
    # Download Bible data for 3 languages
    languages = ["english", "spanish", "italian"]
    
    downloader = BibleDownloader("data/raw")
    download_results = downloader.download_all(languages)
    
    successful_languages = [lang for lang, success in download_results.items() if success]
    logger.info(f"Successfully downloaded: {successful_languages}")
    
    # Preprocess the data
    preprocessor = TextPreprocessor()
    preprocessing_stats = preprocessor.preprocess_all_languages(
        input_dir="data/raw",
        output_dir="data/processed", 
        languages=successful_languages
    )
    
    for lang, stats in preprocessing_stats.items():
        logger.info(f"{lang}: {stats['num_sentences']} sentences, {stats['num_words']} words")
    
    # Step 2: Train embeddings
    logger.info("\\nStep 2: Training embeddings...")
    
    trainer = EmbeddingTrainer("data/embeddings")
    
    # Train Word2Vec embeddings with small parameters for this example
    models = trainer.train_all_languages(
        input_dir="data/processed",
        languages=successful_languages,
        model_type="word2vec",
        vector_size=100,  # Use 100-dimensional vectors
        window=5,         # Context window of 5 words
        min_count=5,      # Ignore words appearing less than 5 times
        epochs=50         # 50 training epochs
    )
    
    # Show vocabulary sizes
    for lang, model in models.items():
        vocab_size = len(model.wv) if model else 0
        logger.info(f"{lang} vocabulary size: {vocab_size}")
    
    # Step 3: Load embeddings and compute distances
    logger.info("\\nStep 3: Computing linguistic distances...")
    
    loader = EmbeddingLoader("data/embeddings")
    
    # Load embeddings as numpy arrays for distance computation
    embeddings_dict = {}
    for language in successful_languages:
        try:
            embeddings = loader.load_embeddings(language, "word2vec")
            
            # Convert to numpy array
            import numpy as np
            words = list(embeddings.keys())
            embeddings_array = np.array([embeddings[word] for word in words])
            embeddings_dict[language] = embeddings_array
            
            logger.info(f"Loaded {len(embeddings)} embeddings for {language}")
            
        except Exception as e:
            logger.error(f"Failed to load embeddings for {language}: {e}")
    
    # Compute cosine similarity distances
    cosine_metrics = CosineSimilarityMetrics()
    
    # Centroid-based distance
    centroid_distances = cosine_metrics.compute_similarity_matrix(
        embeddings_dict, metric='centroid_distance'
    )
    
    # Average pairwise similarity
    average_similarities = cosine_metrics.compute_similarity_matrix(
        embeddings_dict, metric='average_similarity'
    )
    
    # Step 4: Display results
    logger.info("\\nStep 4: Results...")
    
    logger.info("\\nCentroid-based cosine distances:")
    for lang1 in successful_languages:
        for lang2 in successful_languages:
            if lang1 < lang2:  # Show each pair once
                distance = centroid_distances[lang1][lang2]
                logger.info(f"  {lang1} - {lang2}: {distance:.4f}")
    
    logger.info("\\nAverage pairwise similarities:")
    for lang1 in successful_languages:
        for lang2 in successful_languages:
            if lang1 < lang2:
                similarity = average_similarities[lang1][lang2]
                logger.info(f"  {lang1} - {lang2}: {similarity:.4f}")
    
    # Step 5: Create visualizations
    logger.info("\\nStep 5: Creating visualizations...")
    
    try:
        # Create visualization directory
        vis_dir = Path("visualizations")
        vis_dir.mkdir(exist_ok=True)
        
        # Plot centroid distances
        fig1 = plot_distance_matrix(
            centroid_distances,
            title="Centroid-based Cosine Distances",
            save_path="visualizations/centroid_distances.png"
        )
        
        # Plot average similarities  
        fig2 = plot_distance_matrix(
            average_similarities,
            title="Average Pairwise Similarities",
            save_path="visualizations/average_similarities.png",
            colormap='viridis_r'  # Reverse colormap for similarity
        )
        
        logger.info("Visualizations saved to 'visualizations/' directory")
        
        # Close figures to free memory
        import matplotlib.pyplot as plt
        if fig1:
            plt.close(fig1)
        if fig2:
            plt.close(fig2)
        
    except Exception as e:
        logger.error(f"Visualization failed: {e}")
    
    # Step 6: Save results
    logger.info("\\nStep 6: Saving results...")
    
    results = {
        'languages': successful_languages,
        'embedding_stats': {
            lang: {
                'vocabulary_size': len(models[lang].wv) if lang in models and models[lang] else 0
            }
            for lang in successful_languages
        },
        'distance_matrices': {
            'centroid_distances': centroid_distances,
            'average_similarities': average_similarities
        }
    }
    
    save_results(results, "basic_example_results.json")
    logger.info("Results saved to 'results/basic_example_results.json'")
    
    # Summary
    logger.info("\\n=== EXAMPLE COMPLETED SUCCESSFULLY! ===")
    logger.info("Files created:")
    logger.info("  - data/raw/: Downloaded Bible texts")
    logger.info("  - data/processed/: Preprocessed texts")
    logger.info("  - data/embeddings/: Trained word embeddings")
    logger.info("  - visualizations/: Distance matrix heatmaps")
    logger.info("  - results/basic_example_results.json: Analysis results")


if __name__ == "__main__":
    main()