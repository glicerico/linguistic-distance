#!/usr/bin/env python3
"""
Advanced analysis example for the linguistic distance library.

This example demonstrates:
1. Training multiple embedding types
2. Using different distance metrics
3. Performing embedding space alignment
4. Comprehensive evaluation and comparison
5. Creating detailed visualizations and reports
"""

import sys
from pathlib import Path
import logging
import time
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import all components
from embeddings.trainer import EmbeddingTrainer
from embeddings.loader import EmbeddingLoader
from distance.earth_movers import EarthMoversDistance
from distance.cosine_based import CosineSimilarityMetrics
from distance.geometric import GeometricDistances
from alignment.procrustes import ProcrustesAlignment
from alignment.linear_mapping import LinearMapping
from utils.visualization import VisualizationUtils
from utils.io import IOUtils

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train_multiple_embedding_types(languages, input_dir="data/processed"):
    """Train both Word2Vec and FastText embeddings."""
    logger.info("Training multiple embedding types...")
    
    trainer = EmbeddingTrainer("data/embeddings")
    
    # Train Word2Vec embeddings
    logger.info("Training Word2Vec embeddings...")
    word2vec_models = trainer.train_all_languages(
        input_dir=input_dir,
        languages=languages,
        model_type="word2vec",
        vector_size=150,
        window=5,
        min_count=3,
        epochs=100,
        sg=0  # CBOW
    )
    
    # Train FastText embeddings
    logger.info("Training FastText embeddings...")
    fasttext_models = trainer.train_all_languages(
        input_dir=input_dir,
        languages=languages,
        model_type="fasttext",
        vector_size=150,
        window=5,
        min_count=3,
        epochs=100,
        sg=0,  # CBOW
        min_n=3,
        max_n=6
    )
    
    return {
        'word2vec': word2vec_models,
        'fasttext': fasttext_models
    }


def comprehensive_distance_analysis(embeddings_dict, sample_size=200):
    """Perform comprehensive distance analysis using multiple methods."""
    logger.info("Performing comprehensive distance analysis...")
    
    results = {}
    
    # 1. Cosine-based metrics
    logger.info("Computing cosine-based metrics...")
    cosine_metrics = CosineSimilarityMetrics()
    
    results['cosine_centroid'] = cosine_metrics.compute_similarity_matrix(
        embeddings_dict, metric='centroid_distance'
    )
    
    results['cosine_average'] = cosine_metrics.compute_similarity_matrix(
        embeddings_dict, metric='average_similarity'
    )
    
    # 2. Geometric distances
    logger.info("Computing geometric distances...")
    geometric = GeometricDistances()
    
    results['hausdorff'] = geometric.compute_distance_matrix(
        embeddings_dict, metric='hausdorff', sample_size=sample_size
    )
    
    results['chamfer'] = geometric.compute_distance_matrix(
        embeddings_dict, metric='chamfer', sample_size=sample_size
    )
    
    # 3. Earth Mover's Distance (approximation for speed)
    logger.info("Computing Earth Mover's Distance...")
    try:
        emd = EarthMoversDistance(method='approximation')
        results['earth_movers'] = emd.compute_emd_matrix(embeddings_dict)
    except Exception as e:
        logger.warning(f"EMD computation failed: {e}")
        results['earth_movers'] = {}
    
    return results


def alignment_analysis(languages, model_type="word2vec"):
    """Perform embedding space alignment analysis."""
    logger.info("Performing alignment analysis...")
    
    loader = EmbeddingLoader("data/embeddings")
    alignment_results = {}
    
    # Test different alignment methods
    alignment_methods = ['procrustes', 'linear', 'orthogonal']
    
    for method in alignment_methods:
        logger.info(f"Testing {method} alignment...")
        method_results = {}
        
        for i, lang1 in enumerate(languages):
            method_results[lang1] = {}
            for j, lang2 in enumerate(languages):
                if i == j:
                    method_results[lang1][lang2] = 0.0
                elif lang2 in method_results and lang1 in method_results[lang2]:
                    method_results[lang1][lang2] = method_results[lang2][lang1]
                else:
                    try:
                        # Get aligned embeddings
                        emb1, emb2, common_words = loader.align_embeddings(lang1, lang2, model_type)
                        
                        if len(common_words) < 10:
                            method_results[lang1][lang2] = float('inf')
                            continue
                        
                        # Apply alignment and compute error
                        if method == 'procrustes':
                            alignment = ProcrustesAlignment()
                            aligned_emb1 = alignment.fit_transform(emb1, emb2)
                            eval_metrics = alignment.evaluate_alignment(emb1, emb2)
                            error = eval_metrics['mse']
                        else:
                            mapping = LinearMapping()
                            aligned_emb1 = mapping.fit_transform(emb1, emb2, method=method)
                            eval_metrics = mapping.evaluate_alignment(emb1, emb2)
                            error = eval_metrics['mse']
                        
                        method_results[lang1][lang2] = error
                        
                    except Exception as e:
                        logger.warning(f"Alignment failed for {lang1}-{lang2}: {e}")
                        method_results[lang1][lang2] = float('inf')
        
        alignment_results[f'{method}_alignment'] = method_results
    
    return alignment_results


def create_comprehensive_visualizations(all_results, languages):
    """Create comprehensive visualizations."""
    logger.info("Creating comprehensive visualizations...")
    
    viz = VisualizationUtils()
    viz_dir = Path("advanced_visualizations")
    viz_dir.mkdir(exist_ok=True)
    
    # 1. Individual distance matrix heatmaps
    for method_name, distance_matrix in all_results['distances'].items():
        if distance_matrix:
            try:
                title = f"{method_name.replace('_', ' ').title()} Distance Matrix"
                save_path = viz_dir / f"heatmap_{method_name}.png"
                
                fig = viz.plot_distance_matrix(
                    distance_matrix,
                    title=title,
                    save_path=str(save_path)
                )
                
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
            except Exception as e:
                logger.warning(f"Failed to create heatmap for {method_name}: {e}")
    
    # 2. Method comparison plot
    try:
        # Select a few methods for comparison
        comparison_methods = {}
        for method in ['cosine_centroid', 'hausdorff', 'chamfer']:
            if method in all_results['distances'] and all_results['distances'][method]:
                comparison_methods[method] = all_results['distances'][method]
        
        if len(comparison_methods) > 1:
            fig = viz.plot_distance_comparison(
                comparison_methods,
                title="Distance Method Comparison",
                save_path=str(viz_dir / "method_comparison.png")
            )
            
            if fig:
                import matplotlib.pyplot as plt
                plt.close(fig)
                
    except Exception as e:
        logger.warning(f"Failed to create comparison plot: {e}")
    
    # 3. Dendrograms
    for method_name, distance_matrix in all_results['distances'].items():
        if distance_matrix and 'similarity' not in method_name.lower():
            try:
                title = f"{method_name.replace('_', ' ').title()} Dendrogram"
                save_path = viz_dir / f"dendrogram_{method_name}.png"
                
                fig = viz.plot_dendrogram(
                    distance_matrix,
                    title=title,
                    save_path=str(save_path)
                )
                
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
            except Exception as e:
                logger.warning(f"Failed to create dendrogram for {method_name}: {e}")
    
    # 4. Alignment comparison if available
    if 'alignment' in all_results:
        try:
            alignment_methods = all_results['alignment']
            if len(alignment_methods) > 1:
                fig = viz.plot_distance_comparison(
                    alignment_methods,
                    title="Alignment Method Comparison",
                    save_path=str(viz_dir / "alignment_comparison.png")
                )
                
                if fig:
                    import matplotlib.pyplot as plt
                    plt.close(fig)
                    
        except Exception as e:
            logger.warning(f"Failed to create alignment comparison: {e}")


def create_detailed_report(all_results, languages, runtime_info):
    """Create a detailed analysis report."""
    logger.info("Creating detailed analysis report...")
    
    io_utils = IOUtils("advanced_results")
    
    # Save detailed results
    io_utils.save_results(all_results, "comprehensive_analysis.json")
    
    # Create summary report
    report_path = Path("advanced_results") / "analysis_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("COMPREHENSIVE LINGUISTIC DISTANCE ANALYSIS REPORT\\n")
        f.write("=" * 60 + "\\n\\n")
        
        # Basic info
        f.write(f"Languages analyzed: {', '.join(languages)}\\n")
        f.write(f"Total runtime: {runtime_info['total_time']:.2f} seconds\\n")
        f.write(f"Embedding training time: {runtime_info['embedding_time']:.2f} seconds\\n")
        f.write(f"Distance computation time: {runtime_info['distance_time']:.2f} seconds\\n\\n")
        
        # Distance methods summary
        f.write("DISTANCE METHODS ANALYZED:\\n")
        f.write("-" * 30 + "\\n")
        
        for method_name, matrix in all_results['distances'].items():
            if matrix:
                # Calculate statistics
                all_distances = []
                for lang1 in languages:
                    for lang2 in languages:
                        if lang1 != lang2 and lang1 in matrix and lang2 in matrix[lang1]:
                            dist = matrix[lang1][lang2]
                            if np.isfinite(dist):
                                all_distances.append(dist)
                
                if all_distances:
                    f.write(f"\\n{method_name.upper()}:\\n")
                    f.write(f"  Mean distance: {np.mean(all_distances):.6f}\\n")
                    f.write(f"  Std deviation: {np.std(all_distances):.6f}\\n")
                    f.write(f"  Min distance: {np.min(all_distances):.6f}\\n")
                    f.write(f"  Max distance: {np.max(all_distances):.6f}\\n")
                    
                    # Most similar/dissimilar pairs
                    min_dist = np.min(all_distances)
                    max_dist = np.max(all_distances)
                    
                    for lang1 in languages:
                        for lang2 in languages:
                            if (lang1 != lang2 and lang1 in matrix and 
                                lang2 in matrix[lang1]):
                                dist = matrix[lang1][lang2]
                                if abs(dist - min_dist) < 1e-10:
                                    f.write(f"  Most similar: {lang1} - {lang2}\\n")
                                    break
                        else:
                            continue
                        break
                    
                    for lang1 in languages:
                        for lang2 in languages:
                            if (lang1 != lang2 and lang1 in matrix and 
                                lang2 in matrix[lang1]):
                                dist = matrix[lang1][lang2]
                                if abs(dist - max_dist) < 1e-10:
                                    f.write(f"  Most dissimilar: {lang1} - {lang2}\\n")
                                    break
                        else:
                            continue
                        break
        
        # Alignment results if available
        if 'alignment' in all_results:
            f.write("\\n\\nALIGNMENT ANALYSIS:\\n")
            f.write("-" * 20 + "\\n")
            
            for method_name, matrix in all_results['alignment'].items():
                if matrix:
                    f.write(f"\\n{method_name.upper()}:\\n")
                    
                    # Find best and worst alignments
                    alignment_errors = []
                    for lang1 in languages:
                        for lang2 in languages:
                            if (lang1 != lang2 and lang1 in matrix and 
                                lang2 in matrix[lang1]):
                                error = matrix[lang1][lang2]
                                if np.isfinite(error):
                                    alignment_errors.append(error)
                    
                    if alignment_errors:
                        f.write(f"  Mean alignment error: {np.mean(alignment_errors):.6f}\\n")
                        f.write(f"  Best alignment error: {np.min(alignment_errors):.6f}\\n")
                        f.write(f"  Worst alignment error: {np.max(alignment_errors):.6f}\\n")
        
        f.write("\\n\\nFILES GENERATED:\\n")
        f.write("-" * 15 + "\\n")
        f.write("  - advanced_results/comprehensive_analysis.json\\n")
        f.write("  - advanced_visualizations/*.png\\n")
        f.write("  - advanced_results/analysis_report.txt\\n")
    
    logger.info(f"Detailed report saved to {report_path}")


def main():
    """Run advanced analysis example."""
    start_time = time.time()
    
    logger.info("=== LINGUISTIC DISTANCE ANALYSIS - ADVANCED EXAMPLE ===")
    
    # Configuration
    languages = ["english", "spanish", "german", "italian"]
    sample_size = 150  # For expensive computations
    
    runtime_info = {}
    
    try:
        # Step 1: Train multiple embedding types
        logger.info("\\nStep 1: Training multiple embedding types...")
        embedding_start = time.time()
        
        embedding_models = train_multiple_embedding_types(languages)
        runtime_info['embedding_time'] = time.time() - embedding_start
        
        # Step 2: Load embeddings for analysis
        logger.info("\\nStep 2: Loading embeddings for analysis...")
        loader = EmbeddingLoader("data/embeddings")
        
        # We'll focus on Word2Vec for the main analysis
        embeddings_dict = {}
        for language in languages:
            try:
                embeddings = loader.load_embeddings(language, "word2vec")
                words = list(embeddings.keys())
                embeddings_array = np.array([embeddings[word] for word in words])
                embeddings_dict[language] = embeddings_array
                logger.info(f"Loaded {len(embeddings)} word2vec embeddings for {language}")
            except Exception as e:
                logger.error(f"Failed to load embeddings for {language}: {e}")
        
        available_languages = list(embeddings_dict.keys())
        
        # Step 3: Comprehensive distance analysis
        logger.info("\\nStep 3: Comprehensive distance analysis...")
        distance_start = time.time()
        
        distance_results = comprehensive_distance_analysis(embeddings_dict, sample_size)
        runtime_info['distance_time'] = time.time() - distance_start
        
        # Step 4: Alignment analysis
        logger.info("\\nStep 4: Alignment analysis...")
        alignment_results = alignment_analysis(available_languages)
        
        # Combine all results
        all_results = {
            'metadata': {
                'languages': available_languages,
                'sample_size': sample_size,
                'embedding_types': ['word2vec', 'fasttext'],
                'analysis_timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'distances': distance_results,
            'alignment': alignment_results
        }
        
        # Step 5: Create visualizations
        logger.info("\\nStep 5: Creating comprehensive visualizations...")
        create_comprehensive_visualizations(all_results, available_languages)
        
        # Step 6: Create detailed report
        runtime_info['total_time'] = time.time() - start_time
        logger.info("\\nStep 6: Creating detailed analysis report...")
        create_detailed_report(all_results, available_languages, runtime_info)
        
        # Summary
        logger.info("\\n" + "=" * 60)
        logger.info("ADVANCED ANALYSIS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        
        logger.info(f"Languages analyzed: {len(available_languages)}")
        logger.info(f"Distance methods: {len(distance_results)}")
        logger.info(f"Alignment methods: {len(alignment_results)}")
        logger.info(f"Total runtime: {runtime_info['total_time']:.1f} seconds")
        
        logger.info("\\nOutput directories:")
        logger.info("  - advanced_visualizations/: Comprehensive plots and charts")
        logger.info("  - advanced_results/: Detailed analysis results and reports")
        
    except Exception as e:
        logger.error(f"Advanced analysis failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())