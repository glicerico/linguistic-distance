#!/usr/bin/env python3
"""Visualize linguistic distance analysis results."""

import sys
import argparse
from pathlib import Path
import logging
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.io import IOUtils
from utils.visualization import VisualizationUtils
from embeddings.loader import EmbeddingLoader


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function for visualizing results."""
    parser = argparse.ArgumentParser(description="Visualize linguistic distance analysis results")
    
    parser.add_argument(
        "--results-file", 
        required=True,
        help="JSON file containing distance computation results"
    )
    parser.add_argument(
        "--results-dir", 
        default="results",
        help="Directory containing results files"
    )
    parser.add_argument(
        "--output-dir", 
        default="visualizations",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--embeddings-dir", 
        default="data/embeddings",
        help="Directory containing embeddings for 2D visualization"
    )
    parser.add_argument(
        "--plot-types", 
        nargs="+",
        default=["heatmaps", "dendrograms", "embeddings"],
        choices=["heatmaps", "dendrograms", "embeddings", "comparisons"],
        help="Types of plots to generate"
    )
    parser.add_argument(
        "--embedding-viz-method", 
        default="pca",
        choices=["pca", "tsne", "mds"],
        help="Method for embedding visualization"
    )
    parser.add_argument(
        "--model-type", 
        default="word2vec",
        help="Type of embeddings to visualize"
    )
    parser.add_argument(
        "--dpi", 
        type=int, 
        default=300,
        help="DPI for saved figures"
    )
    parser.add_argument(
        "--format", 
        default="png",
        choices=["png", "jpg", "pdf", "svg"],
        help="Format for saved figures"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    io_utils = IOUtils(args.results_dir)
    viz = VisualizationUtils()
    
    try:
        results = io_utils.load_results(args.results_file)
        logger.info(f"Loaded results from {args.results_file}")
    except Exception as e:
        logger.error(f"Failed to load results: {e}")
        return 1
        
    # Handle nested structure - check if results are nested under 'results' key
    if 'results' in results and 'distance_matrices' in results['results']:
        distance_matrices = results['results']['distance_matrices']
        metadata = results['results'].get('metadata', {})
    elif 'distance_matrices' in results:
        distance_matrices = results['distance_matrices']
        metadata = results.get('metadata', {})
    else:
        logger.error("No distance matrices found in results file")
        return 1
    languages = metadata.get('languages', [])
    
    logger.info(f"Visualizing results for {len(languages)} languages: {languages}")
    logger.info(f"Available distance methods: {list(distance_matrices.keys())}")
    
    # Generate heatmaps
    if 'heatmaps' in args.plot_types:
        logger.info("Generating heatmap visualizations...")
        
        for method_name, matrix in distance_matrices.items():
            if matrix:  # Only process non-empty matrices
                try:
                    title = f"{method_name.replace('_', ' ').title()} Distance Matrix"
                    filename = output_dir / f"heatmap_{method_name}.{args.format}"
                    
                    fig = viz.plot_distance_matrix(
                        matrix, 
                        title=title,
                        save_path=str(filename)
                    )
                    plt.close(fig)
                    
                    logger.info(f"Saved heatmap: {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to create heatmap for {method_name}: {e}")
    
    # Generate dendrograms
    if 'dendrograms' in args.plot_types:
        logger.info("Generating dendrogram visualizations...")
        
        for method_name, matrix in distance_matrices.items():
            if matrix:
                try:
                    # Skip similarity matrices (convert to distance if needed)
                    if 'similarity' in method_name.lower():
                        # Convert similarity to distance
                        distance_matrix = {}
                        for lang1, similarities in matrix.items():
                            distance_matrix[lang1] = {}
                            for lang2, sim in similarities.items():
                                # Convert similarity to distance (1 - similarity)
                                if lang1 == lang2:
                                    distance_matrix[lang1][lang2] = 0.0
                                else:
                                    distance_matrix[lang1][lang2] = max(0, 1 - sim)
                        matrix_for_dendrogram = distance_matrix
                    else:
                        matrix_for_dendrogram = matrix
                    
                    title = f"{method_name.replace('_', ' ').title()} Dendrogram"
                    filename = output_dir / f"dendrogram_{method_name}.{args.format}"
                    
                    fig = viz.plot_dendrogram(
                        matrix_for_dendrogram,
                        title=title,
                        save_path=str(filename)
                    )
                    plt.close(fig)
                    
                    logger.info(f"Saved dendrogram: {filename}")
                    
                except Exception as e:
                    logger.error(f"Failed to create dendrogram for {method_name}: {e}")
    
    # Generate embedding visualizations
    if 'embeddings' in args.plot_types:
        logger.info("Generating embedding visualizations...")
        
        try:
            loader = EmbeddingLoader(args.embeddings_dir)
            embeddings_dict = {}
            
            for language in languages:
                try:
                    embeddings = loader.load_embeddings(language, args.model_type)
                    # Convert to numpy array
                    import numpy as np
                    words = list(embeddings.keys())
                    embeddings_array = np.array([embeddings[word] for word in words])
                    embeddings_dict[language] = embeddings_array
                except Exception as e:
                    logger.warning(f"Could not load embeddings for {language}: {e}")
                    
            if embeddings_dict:
                title = f"{args.model_type.title()} Embeddings ({args.embedding_viz_method.upper()})"
                filename = output_dir / f"embeddings_{args.model_type}_{args.embedding_viz_method}.{args.format}"
                
                fig = viz.plot_embeddings_2d(
                    embeddings_dict,
                    method=args.embedding_viz_method,
                    title=title,
                    save_path=str(filename)
                )
                plt.close(fig)
                
                logger.info(f"Saved embedding visualization: {filename}")
            else:
                logger.warning("No embeddings could be loaded for visualization")
                
        except Exception as e:
            logger.error(f"Failed to create embedding visualization: {e}")
    
    # Generate comparison plots
    if 'comparisons' in args.plot_types and len(distance_matrices) > 1:
        logger.info("Generating comparison visualizations...")
        
        try:
            # Select up to 4 methods for comparison (to fit in subplot grid)
            comparison_methods = list(distance_matrices.keys())[:4]
            comparison_matrices = {method: distance_matrices[method] 
                                 for method in comparison_methods 
                                 if distance_matrices[method]}
            
            if len(comparison_matrices) > 1:
                title = "Distance Method Comparison"
                filename = output_dir / f"comparison_methods.{args.format}"
                
                fig = viz.plot_distance_comparison(
                    comparison_matrices,
                    title=title,
                    save_path=str(filename)
                )
                plt.close(fig)
                
                logger.info(f"Saved comparison visualization: {filename}")
            else:
                logger.warning("Not enough valid distance matrices for comparison")
                
        except Exception as e:
            logger.error(f"Failed to create comparison visualization: {e}")
    
    # Generate embedding statistics plots if available
    if 'embedding_stats' in results:
        logger.info("Generating embedding statistics visualizations...")
        
        try:
            stats_dict = results['embedding_stats']
            title = f"{args.model_type.title()} Embedding Statistics"
            filename = output_dir / f"embedding_stats_{args.model_type}.{args.format}"
            
            # Select relevant metrics for visualization
            metrics = ['vocabulary_size', 'mean_vector_norm', 'std_vector_norm']
            
            fig = viz.plot_embedding_statistics(
                stats_dict,
                metrics=metrics,
                title=title,
                save_path=str(filename)
            )
            plt.close(fig)
            
            logger.info(f"Saved embedding statistics: {filename}")
            
        except Exception as e:
            logger.error(f"Failed to create embedding statistics visualization: {e}")
    
    # Create summary report
    logger.info("Creating summary report...")
    try:
        report_file = output_dir / "visualization_summary.txt"
        
        with open(report_file, 'w') as f:
            f.write("LINGUISTIC DISTANCE VISUALIZATION SUMMARY\\n")
            f.write("=" * 50 + "\\n\\n")
            
            f.write(f"Results file: {args.results_file}\\n")
            f.write(f"Languages analyzed: {', '.join(languages)}\\n")
            f.write(f"Distance methods: {', '.join(distance_matrices.keys())}\\n")
            f.write(f"Visualization types generated: {', '.join(args.plot_types)}\\n\\n")
            
            # List generated files
            f.write("Generated visualizations:\\n")
            f.write("-" * 30 + "\\n")
            
            for viz_file in sorted(output_dir.glob(f"*.{args.format}")):
                file_size = viz_file.stat().st_size / 1024  # KB
                f.write(f"  {viz_file.name} ({file_size:.1f} KB)\\n")
                
        logger.info(f"Created summary report: {report_file}")
        
    except Exception as e:
        logger.error(f"Failed to create summary report: {e}")
    
    logger.info(f"Visualization complete! Results saved to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())