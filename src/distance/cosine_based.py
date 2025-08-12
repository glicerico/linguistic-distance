"""Cosine similarity-based distance metrics for embedding spaces."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import cosine, cdist
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
import logging


class CosineSimilarityMetrics:
    """Compute cosine similarity-based distance metrics between embedding spaces."""
    
    def __init__(self):
        """Initialize the cosine similarity metrics calculator."""
        self.logger = logging.getLogger(__name__)
        
    def compute_centroid_distance(self, 
                                 embeddings1: np.ndarray, 
                                 embeddings2: np.ndarray,
                                 weights1: Optional[np.ndarray] = None,
                                 weights2: Optional[np.ndarray] = None) -> float:
        """Compute cosine distance between weighted centroids.
        
        Args:
            embeddings1: First embedding matrix (n1 x d)
            embeddings2: Second embedding matrix (n2 x d)
            weights1: Weights for first embeddings
            weights2: Weights for second embeddings
            
        Returns:
            Cosine distance between centroids
        """
        # Compute weighted centroids
        if weights1 is not None:
            weights1 = weights1 / np.sum(weights1)
            centroid1 = np.average(embeddings1, axis=0, weights=weights1)
        else:
            centroid1 = np.mean(embeddings1, axis=0)
            
        if weights2 is not None:
            weights2 = weights2 / np.sum(weights2)
            centroid2 = np.average(embeddings2, axis=0, weights=weights2)
        else:
            centroid2 = np.mean(embeddings2, axis=0)
            
        # Compute cosine distance
        return float(cosine(centroid1, centroid2))
        
    def compute_average_similarity(self, 
                                  embeddings1: np.ndarray, 
                                  embeddings2: np.ndarray,
                                  method: str = 'all_pairs') -> float:
        """Compute average cosine similarity between embedding sets.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            method: Method for computing average ('all_pairs', 'nearest_neighbor', 'random_sample')
            
        Returns:
            Average cosine similarity
        """
        if method == 'all_pairs':
            return self._average_all_pairs(embeddings1, embeddings2)
        elif method == 'nearest_neighbor':
            return self._average_nearest_neighbor(embeddings1, embeddings2)
        elif method == 'random_sample':
            return self._average_random_sample(embeddings1, embeddings2)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _average_all_pairs(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Compute average similarity across all pairs."""
        similarities = cosine_similarity(embeddings1, embeddings2)
        return float(np.mean(similarities))
        
    def _average_nearest_neighbor(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> float:
        """Compute average similarity using nearest neighbors."""
        similarities = cosine_similarity(embeddings1, embeddings2)
        
        # For each point in embeddings1, find its nearest neighbor in embeddings2
        nearest_similarities = np.max(similarities, axis=1)
        
        return float(np.mean(nearest_similarities))
        
    def _average_random_sample(self, 
                              embeddings1: np.ndarray, 
                              embeddings2: np.ndarray,
                              n_samples: int = 1000) -> float:
        """Compute average similarity using random sampling."""
        n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
        
        # Sample random pairs
        n_samples = min(n_samples, n1 * n2)
        i_indices = np.random.choice(n1, n_samples, replace=True)
        j_indices = np.random.choice(n2, n_samples, replace=True)
        
        similarities = []
        for i, j in zip(i_indices, j_indices):
            sim = np.dot(embeddings1[i], embeddings2[j]) / (
                np.linalg.norm(embeddings1[i]) * np.linalg.norm(embeddings2[j])
            )
            similarities.append(sim)
            
        return float(np.mean(similarities))
        
    def compute_distribution_similarity(self, 
                                       embeddings1: np.ndarray, 
                                       embeddings2: np.ndarray,
                                       method: str = 'moments') -> Dict[str, float]:
        """Compare similarity of embedding distributions.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            method: Method for distribution comparison ('moments', 'correlation')
            
        Returns:
            Dictionary of distribution similarity metrics
        """
        if method == 'moments':
            return self._moment_based_similarity(embeddings1, embeddings2)
        elif method == 'correlation':
            return self._correlation_based_similarity(embeddings1, embeddings2)
        else:
            raise ValueError(f"Unknown method: {method}")
            
    def _moment_based_similarity(self, 
                                embeddings1: np.ndarray, 
                                embeddings2: np.ndarray) -> Dict[str, float]:
        """Compare distributions using statistical moments."""
        # First and second moments
        mean1, mean2 = np.mean(embeddings1, axis=0), np.mean(embeddings2, axis=0)
        var1, var2 = np.var(embeddings1, axis=0), np.var(embeddings2, axis=0)
        
        # Cosine similarity between means
        mean_similarity = np.dot(mean1, mean2) / (np.linalg.norm(mean1) * np.linalg.norm(mean2))
        
        # Correlation between variances
        var_correlation, var_p_value = pearsonr(var1, var2)
        
        # KL-divergence approximation (assuming Gaussian)
        kl_div_approx = 0.5 * np.mean(
            np.log(var2 / var1) + var1 / var2 + (mean1 - mean2)**2 / var2 - 1
        )
        
        return {
            'mean_cosine_similarity': float(mean_similarity),
            'variance_correlation': float(var_correlation),
            'variance_correlation_p_value': float(var_p_value),
            'kl_divergence_approximation': float(kl_div_approx),
            'mean_l2_distance': float(np.linalg.norm(mean1 - mean2)),
            'variance_l2_distance': float(np.linalg.norm(var1 - var2))
        }
        
    def _correlation_based_similarity(self, 
                                     embeddings1: np.ndarray, 
                                     embeddings2: np.ndarray) -> Dict[str, float]:
        """Compare distributions using correlation analysis."""
        # Flatten embeddings for correlation analysis
        flat1 = embeddings1.flatten()
        flat2 = embeddings2.flatten()
        
        # Make them the same length by sampling
        min_length = min(len(flat1), len(flat2))
        idx1 = np.random.choice(len(flat1), min_length, replace=False)
        idx2 = np.random.choice(len(flat2), min_length, replace=False)
        
        sample1 = flat1[idx1]
        sample2 = flat2[idx2]
        
        # Compute correlations
        pearson_corr, pearson_p = pearsonr(sample1, sample2)
        spearman_corr, spearman_p = spearmanr(sample1, sample2)
        
        return {
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p)
        }
        
    def compute_subspace_similarity(self, 
                                   embeddings1: np.ndarray, 
                                   embeddings2: np.ndarray,
                                   n_components: int = 10) -> Dict[str, float]:
        """Compare principal subspaces of embedding spaces.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            n_components: Number of principal components to compare
            
        Returns:
            Dictionary of subspace similarity metrics
        """
        from sklearn.decomposition import PCA
        
        # Fit PCA on both embedding sets
        pca1 = PCA(n_components=n_components)
        pca2 = PCA(n_components=n_components)
        
        pca1.fit(embeddings1)
        pca2.fit(embeddings2)
        
        # Get principal components
        components1 = pca1.components_
        components2 = pca2.components_
        
        # Compute similarities between principal components
        component_similarities = cosine_similarity(components1, components2)
        
        # Principal angles between subspaces
        principal_angles = self._compute_principal_angles(components1.T, components2.T)
        
        return {
            'max_component_similarity': float(np.max(component_similarities)),
            'mean_component_similarity': float(np.mean(component_similarities)),
            'explained_variance_correlation': float(pearsonr(pca1.explained_variance_, pca2.explained_variance_)[0]),
            'mean_principal_angle': float(np.mean(principal_angles)),
            'max_principal_angle': float(np.max(principal_angles)),
            'subspace_distance': float(np.linalg.norm(np.sin(principal_angles)))
        }
        
    def _compute_principal_angles(self, U: np.ndarray, V: np.ndarray) -> np.ndarray:
        """Compute principal angles between two subspaces.
        
        Args:
            U: Orthonormal basis for first subspace
            V: Orthonormal basis for second subspace
            
        Returns:
            Principal angles in radians
        """
        # Ensure orthonormal bases
        U, _ = np.linalg.qr(U)
        V, _ = np.linalg.qr(V)
        
        # Compute SVD of U^T V
        _, s, _ = np.linalg.svd(U.T @ V)
        
        # Clamp singular values to [0, 1] to avoid numerical issues
        s = np.clip(s, 0, 1)
        
        # Principal angles are arccos of singular values
        angles = np.arccos(s)
        
        return angles
        
    def compute_comprehensive_metrics(self, 
                                     embeddings1: np.ndarray, 
                                     embeddings2: np.ndarray,
                                     weights1: Optional[np.ndarray] = None,
                                     weights2: Optional[np.ndarray] = None) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute comprehensive cosine-based similarity metrics.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Optional weights for first embeddings
            weights2: Optional weights for second embeddings
            
        Returns:
            Dictionary of all computed metrics
        """
        metrics = {}
        
        # Centroid distance
        metrics['centroid_cosine_distance'] = self.compute_centroid_distance(
            embeddings1, embeddings2, weights1, weights2
        )
        
        # Average similarities
        metrics['average_similarity_all_pairs'] = self.compute_average_similarity(
            embeddings1, embeddings2, 'all_pairs'
        )
        metrics['average_similarity_nearest_neighbor'] = self.compute_average_similarity(
            embeddings1, embeddings2, 'nearest_neighbor'
        )
        
        # Distribution similarities
        metrics['distribution_moments'] = self.compute_distribution_similarity(
            embeddings1, embeddings2, 'moments'
        )
        metrics['distribution_correlation'] = self.compute_distribution_similarity(
            embeddings1, embeddings2, 'correlation'
        )
        
        # Subspace similarity
        try:
            n_components = min(10, embeddings1.shape[1], embeddings2.shape[1])
            metrics['subspace_similarity'] = self.compute_subspace_similarity(
                embeddings1, embeddings2, n_components
            )
        except Exception as e:
            self.logger.warning(f"Subspace similarity computation failed: {e}")
            metrics['subspace_similarity'] = {}
            
        return metrics
        
    def compute_similarity_matrix(self, 
                                 embeddings_dict: Dict[str, np.ndarray],
                                 weights_dict: Optional[Dict[str, np.ndarray]] = None,
                                 metric: str = 'centroid_distance') -> Dict[str, Dict[str, float]]:
        """Compute similarity matrix between all pairs of embedding sets.
        
        Args:
            embeddings_dict: Dictionary mapping language names to embeddings
            weights_dict: Optional dictionary mapping language names to weights
            metric: Which metric to use for the matrix
            
        Returns:
            Dictionary of dictionaries with pairwise similarity values
        """
        languages = list(embeddings_dict.keys())
        similarity_matrix = {}
        
        if weights_dict is None:
            weights_dict = {}
            
        for i, lang1 in enumerate(languages):
            similarity_matrix[lang1] = {}
            for j, lang2 in enumerate(languages):
                if i == j:
                    similarity_matrix[lang1][lang2] = 0.0 if 'distance' in metric else 1.0
                elif lang2 in similarity_matrix and lang1 in similarity_matrix[lang2]:
                    # Use symmetry
                    similarity_matrix[lang1][lang2] = similarity_matrix[lang2][lang1]
                else:
                    weights1 = weights_dict.get(lang1, None)
                    weights2 = weights_dict.get(lang2, None)
                    
                    try:
                        if metric == 'centroid_distance':
                            value = self.compute_centroid_distance(
                                embeddings_dict[lang1], 
                                embeddings_dict[lang2],
                                weights1, weights2
                            )
                        elif metric == 'average_similarity':
                            value = self.compute_average_similarity(
                                embeddings_dict[lang1], 
                                embeddings_dict[lang2],
                                'all_pairs'
                            )
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                            
                        similarity_matrix[lang1][lang2] = value
                        self.logger.info(f"Cosine {metric} {lang1}-{lang2}: {value:.6f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to compute {metric} for {lang1}-{lang2}: {e}")
                        similarity_matrix[lang1][lang2] = float('inf') if 'distance' in metric else 0.0
                        
        return similarity_matrix