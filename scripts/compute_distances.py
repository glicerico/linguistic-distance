#!/usr/bin/env python3
"""Compute linguistic distances between embedding spaces."""

import sys
import argparse
from pathlib import Path
import logging
import json
import time
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.loader import EmbeddingLoader
from distance.earth_movers import EarthMoversDistance
from distance.cosine_based import CosineSimilarityMetrics
from distance.geometric import GeometricDistances
from alignment.linear_mapping import LinearMapping
from alignment.procrustes import ProcrustesAlignment
from utils.io import IOUtils


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def compute_all_distances(embeddings_dict: Dict[str, Any], 
                         distance_methods: list,
                         sample_size: int = 100) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute distances using all specified methods.
    
    Args:
        embeddings_dict: Dictionary of embeddings by language
        distance_methods: List of distance methods to compute
        sample_size: Sample size for expensive computations
        
    Returns:
        Nested dictionary of distance matrices by method
    """
    results = {}
    logger = logging.getLogger(__name__)
    
    if 'earth_movers' in distance_methods:
        logger.info("Computing Earth Mover's Distance...")
        emd = EarthMoversDistance(method='approximation')  # Use approximation for speed
        try:
            results['earth_movers_distance'] = emd.compute_emd_matrix(embeddings_dict)
        except Exception as e:
            logger.error(f"Earth Mover's Distance computation failed: {e}")
            results['earth_movers_distance'] = {}
    
    if 'cosine_centroid' in distance_methods:
        logger.info("Computing cosine centroid distances...")
        cosine_metrics = CosineSimilarityMetrics()
        try:
            results['cosine_centroid_distance'] = cosine_metrics.compute_similarity_matrix(
                embeddings_dict, metric='centroid_distance'
            )
        except Exception as e:
            logger.error(f"Cosine centroid distance computation failed: {e}")
            results['cosine_centroid_distance'] = {}
    
    if 'cosine_average' in distance_methods:
        logger.info("Computing average cosine similarities...")
        cosine_metrics = CosineSimilarityMetrics()
        try:
            results['cosine_average_similarity'] = cosine_metrics.compute_similarity_matrix(
                embeddings_dict, metric='average_similarity'
            )
        except Exception as e:
            logger.error(f"Average cosine similarity computation failed: {e}")
            results['cosine_average_similarity'] = {}
    
    if 'hausdorff' in distance_methods:
        logger.info("Computing Hausdorff distances...")
        geometric = GeometricDistances()
        try:
            results['hausdorff_distance'] = geometric.compute_distance_matrix(
                embeddings_dict, metric='hausdorff', sample_size=sample_size
            )
        except Exception as e:
            logger.error(f"Hausdorff distance computation failed: {e}")
            results['hausdorff_distance'] = {}
    
    if 'chamfer' in distance_methods:
        logger.info("Computing Chamfer distances...")
        geometric = GeometricDistances()
        try:
            results['chamfer_distance'] = geometric.compute_distance_matrix(
                embeddings_dict, metric='chamfer', sample_size=sample_size
            )
        except Exception as e:
            logger.error(f"Chamfer distance computation failed: {e}")
            results['chamfer_distance'] = {}
    
    return results


def compute_alignment_distances(embeddings_dict: Dict[str, Any],
                               alignment_methods: list) -> Dict[str, Dict[str, Dict[str, float]]]:
    """Compute distances after alignment.
    
    Args:
        embeddings_dict: Dictionary of embeddings by language
        alignment_methods: List of alignment methods to use
        
    Returns:
        Nested dictionary of alignment-based distance matrices
    """
    results = {}
    logger = logging.getLogger(__name__)
    loader = EmbeddingLoader()
    
    languages = list(embeddings_dict.keys())
    
    for method in alignment_methods:
        logger.info(f"Computing {method} alignment distances...")
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
                        emb1, emb2, common_words = loader.align_embeddings(lang1, lang2)
                        
                        if len(common_words) == 0:
                            logger.warning(f"No common vocabulary between {lang1} and {lang2}")
                            method_results[lang1][lang2] = float('inf')
                            continue
                        
                        # Apply alignment
                        if method == 'procrustes':
                            alignment = ProcrustesAlignment()
                            aligned_emb1 = alignment.fit_transform(emb1, emb2)
                            # Compute MSE after alignment
                            mse = float(((aligned_emb1 - emb2) ** 2).mean())
                            method_results[lang1][lang2] = mse
                        elif method == 'linear':
                            alignment = LinearMapping()
                            aligned_emb1 = alignment.fit_transform(emb1, emb2, method='linear')
                            # Compute MSE after alignment
                            mse = float(((aligned_emb1 - emb2) ** 2).mean())
                            method_results[lang1][lang2] = mse
                        elif method == 'orthogonal':
                            alignment = LinearMapping()
                            aligned_emb1 = alignment.fit_transform(emb1, emb2, method='orthogonal')
                            # Compute MSE after alignment
                            mse = float(((aligned_emb1 - emb2) ** 2).mean())
                            method_results[lang1][lang2] = mse
                            
                        logger.info(f"{method} alignment distance {lang1}-{lang2}: {method_results[lang1][lang2]:.6f}")
                        
                    except Exception as e:
                        logger.error(f"Failed to compute {method} alignment for {lang1}-{lang2}: {e}")
                        method_results[lang1][lang2] = float('inf')
        
        results[f'{method}_alignment_distance'] = method_results
    
    return results


def main():
    """Main function for computing linguistic distances."""
    parser = argparse.ArgumentParser(description="Compute linguistic distances between embedding spaces")
    
    parser.add_argument(
        "--languages", 
        nargs="+", 
        default=["english", "spanish", "german", "italian", "dutch"],
        help="Languages to analyze"
    )
    parser.add_argument(
        "--embeddings-dir", 
        default="data/embeddings",
        help="Directory containing trained embeddings"
    )
    parser.add_argument(
        "--model-type", 
        default="word2vec",
        choices=["word2vec", "fasttext"],
        help="Type of embeddings to use"
    )
    parser.add_argument(
        "--output-dir", 
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--distance-methods", 
        nargs="+",
        default=["cosine_centroid", "cosine_average", "hausdorff"],
        choices=["earth_movers", "cosine_centroid", "cosine_average", "hausdorff", "chamfer"],
        help="Distance methods to compute"
    )
    parser.add_argument(
        "--alignment-methods", 
        nargs="+",
        default=["procrustes"],
        choices=["procrustes", "linear", "orthogonal"],
        help="Alignment methods to use"
    )
    parser.add_argument(
        "--sample-size", 
        type=int, 
        default=100,
        help="Sample size for expensive distance computations"
    )
    parser.add_argument(
        "--skip-alignment", 
        action="store_true",
        help="Skip alignment-based distance computation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Initialize components
    loader = EmbeddingLoader(args.embeddings_dir)
    io_utils = IOUtils(args.output_dir)
    
    # Load embeddings
    logger.info(f"Loading {args.model_type} embeddings for: {args.languages}")
    embeddings_dict = {}
    
    for language in args.languages:
        try:
            embeddings = loader.load_embeddings(language, args.model_type)
            embeddings_dict[language] = embeddings
            logger.info(f"Loaded {len(embeddings)} {language} embeddings")
        except Exception as e:
            logger.error(f"Failed to load embeddings for {language}: {e}")
            
    if not embeddings_dict:
        logger.error("No embeddings could be loaded. Exiting.")
        return 1
        
    available_languages = list(embeddings_dict.keys())
    logger.info(f"Computing distances for {len(available_languages)} languages: {available_languages}")
    
    # Convert embeddings to numpy arrays for distance computation
    embeddings_arrays = {}
    for lang, emb_dict in embeddings_dict.items():
        import numpy as np
        words = list(emb_dict.keys())
        embeddings_arrays[lang] = np.array([emb_dict[word] for word in words])
        
    # Compute direct distances
    start_time = time.time()
    logger.info("Computing direct distance metrics...")
    
    distance_results = compute_all_distances(
        embeddings_arrays, 
        args.distance_methods,
        args.sample_size
    )
    
    direct_time = time.time() - start_time
    logger.info(f"Direct distance computation completed in {direct_time:.2f} seconds")
    
    # Compute alignment-based distances
    alignment_results = {}
    if not args.skip_alignment:
        start_time = time.time()
        logger.info("Computing alignment-based distances...")
        
        try:
            alignment_results = compute_alignment_distances(
                embeddings_dict,
                args.alignment_methods
            )
            alignment_time = time.time() - start_time
            logger.info(f"Alignment-based computation completed in {alignment_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Alignment computation failed: {e}")
    
    # Combine all results
    all_results = {
        'metadata': {
            'languages': available_languages,
            'model_type': args.model_type,
            'distance_methods': args.distance_methods,
            'alignment_methods': args.alignment_methods if not args.skip_alignment else [],
            'sample_size': args.sample_size
        },
        'distance_matrices': {**distance_results, **alignment_results}
    }
    
    # Save results
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = f"linguistic_distances_{args.model_type}_{timestamp}.json"
    
    io_utils.save_results(all_results, results_file)
    logger.info(f"Results saved to {results_file}")
    
    # Create summary
    logger.info("\\nDISTANCE COMPUTATION SUMMARY:")
    logger.info("=" * 50)
    
    for method_name, matrix in all_results['distance_matrices'].items():
        if matrix:  # Only process non-empty matrices
            # Compute basic statistics
            all_distances = []
            language_list = list(matrix.keys())
            
            for lang1 in language_list:
                for lang2 in language_list:
                    if lang1 != lang2:
                        dist = matrix[lang1].get(lang2, float('inf'))
                        if dist != float('inf') and not (dist != dist):  # Not inf and not NaN
                            all_distances.append(dist)
            
            if all_distances:
                import numpy as np
                logger.info(f"\\n{method_name.upper()}:")
                logger.info(f"  Mean: {np.mean(all_distances):.6f}")
                logger.info(f"  Std:  {np.std(all_distances):.6f}")
                logger.info(f"  Min:  {np.min(all_distances):.6f}")
                logger.info(f"  Max:  {np.max(all_distances):.6f}")
                
                # Find most similar and dissimilar pairs
                min_dist = np.min(all_distances)
                max_dist = np.max(all_distances)
                
                for lang1 in language_list:
                    for lang2 in language_list:
                        if lang1 != lang2:
                            dist = matrix[lang1].get(lang2, float('inf'))
                            if abs(dist - min_dist) < 1e-10:
                                logger.info(f"  Most similar: {lang1} - {lang2} ({dist:.6f})")
                                break
                    else:
                        continue
                    break
                
                for lang1 in language_list:
                    for lang2 in language_list:
                        if lang1 != lang2:
                            dist = matrix[lang1].get(lang2, float('inf'))
                            if abs(dist - max_dist) < 1e-10:
                                logger.info(f"  Most dissimilar: {lang1} - {lang2} ({dist:.6f})")
                                break
                    else:
                        continue
                    break
    
    logger.info("\\nDistance computation complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())