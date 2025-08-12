"""Geometric distance measures for embedding spaces."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from scipy.spatial.distance import pdist, squareform
from scipy.stats import wasserstein_distance
from sklearn.manifold import MDS
from sklearn.decomposition import PCA
import logging


class GeometricDistances:
    """Compute geometric distance measures between embedding spaces."""
    
    def __init__(self):
        """Initialize the geometric distance calculator."""
        self.logger = logging.getLogger(__name__)
        
    def hausdorff_distance(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray) -> float:
        """Compute Hausdorff distance between two embedding sets.
        
        Args:
            embeddings1: First embedding matrix (n1 x d)
            embeddings2: Second embedding matrix (n2 x d)
            
        Returns:
            Hausdorff distance
        """
        # Distance from each point in set1 to closest point in set2
        from scipy.spatial.distance import cdist
        
        distances_1_to_2 = cdist(embeddings1, embeddings2)
        min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
        max_min_1_to_2 = np.max(min_distances_1_to_2)
        
        # Distance from each point in set2 to closest point in set1
        distances_2_to_1 = cdist(embeddings2, embeddings1)
        min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
        max_min_2_to_1 = np.max(min_distances_2_to_1)
        
        # Hausdorff distance is the maximum of these two values
        hausdorff_dist = max(max_min_1_to_2, max_min_2_to_1)
        
        return float(hausdorff_dist)
        
    def chamfer_distance(self, 
                        embeddings1: np.ndarray, 
                        embeddings2: np.ndarray) -> float:
        """Compute Chamfer distance between two embedding sets.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            
        Returns:
            Chamfer distance
        """
        from scipy.spatial.distance import cdist
        
        # Distance from each point in set1 to closest point in set2
        distances_1_to_2 = cdist(embeddings1, embeddings2)
        min_distances_1_to_2 = np.min(distances_1_to_2, axis=1)
        
        # Distance from each point in set2 to closest point in set1
        distances_2_to_1 = cdist(embeddings2, embeddings1)
        min_distances_2_to_1 = np.min(distances_2_to_1, axis=1)
        
        # Chamfer distance is the sum of average minimum distances
        chamfer_dist = np.mean(min_distances_1_to_2) + np.mean(min_distances_2_to_1)
        
        return float(chamfer_dist)
        
    def gromov_hausdorff_approximation(self, 
                                      embeddings1: np.ndarray, 
                                      embeddings2: np.ndarray,
                                      sample_size: Optional[int] = None) -> float:
        """Approximate Gromov-Hausdorff distance between embedding spaces.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            sample_size: Number of points to sample for approximation
            
        Returns:
            Approximated Gromov-Hausdorff distance
        """
        # Sample points if sets are too large
        if sample_size is not None:
            if embeddings1.shape[0] > sample_size:
                idx1 = np.random.choice(embeddings1.shape[0], sample_size, replace=False)
                embeddings1 = embeddings1[idx1]
            if embeddings2.shape[0] > sample_size:
                idx2 = np.random.choice(embeddings2.shape[0], sample_size, replace=False)
                embeddings2 = embeddings2[idx2]
                
        # Compute pairwise distance matrices
        dist_matrix1 = squareform(pdist(embeddings1))
        dist_matrix2 = squareform(pdist(embeddings2))
        
        # Find the best correspondence using a simplified approach
        # (True Gromov-Hausdorff is computationally intensive)
        min_distortion = float('inf')
        
        # Try a few random correspondences
        n_trials = min(100, embeddings1.shape[0] * embeddings2.shape[0])
        
        for _ in range(n_trials):
            # Create random correspondence
            if embeddings1.shape[0] <= embeddings2.shape[0]:
                correspondence = np.random.choice(embeddings2.shape[0], embeddings1.shape[0], replace=False)
                dist_matrix2_subset = dist_matrix2[np.ix_(correspondence, correspondence)]
                distortion = np.max(np.abs(dist_matrix1 - dist_matrix2_subset))
            else:
                correspondence = np.random.choice(embeddings1.shape[0], embeddings2.shape[0], replace=False)
                dist_matrix1_subset = dist_matrix1[np.ix_(correspondence, correspondence)]
                distortion = np.max(np.abs(dist_matrix1_subset - dist_matrix2))
                
            min_distortion = min(min_distortion, distortion)
            
        return float(min_distortion)
        
    def shape_context_distance(self, 
                              embeddings1: np.ndarray, 
                              embeddings2: np.ndarray,
                              n_bins_r: int = 5,
                              n_bins_theta: int = 12) -> float:
        """Compute shape context distance between embedding point clouds.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            n_bins_r: Number of radial bins
            n_bins_theta: Number of angular bins
            
        Returns:
            Shape context distance
        """
        def compute_shape_context(points, n_bins_r, n_bins_theta):
            """Compute shape context for a set of points."""
            n_points = points.shape[0]
            contexts = []
            
            for i, point in enumerate(points):
                # Compute relative positions
                relative_positions = points - point
                
                # Convert to polar coordinates (in 2D projection)
                if points.shape[1] > 2:
                    # Project to 2D using PCA
                    pca = PCA(n_components=2)
                    relative_positions_2d = pca.fit_transform(relative_positions)
                else:
                    relative_positions_2d = relative_positions
                    
                distances = np.linalg.norm(relative_positions_2d, axis=1)
                angles = np.arctan2(relative_positions_2d[:, 1], relative_positions_2d[:, 0])
                
                # Create histogram
                max_dist = np.max(distances[distances > 0])
                r_bins = np.logspace(np.log10(max_dist/100), np.log10(max_dist), n_bins_r + 1)
                theta_bins = np.linspace(-np.pi, np.pi, n_bins_theta + 1)
                
                hist, _, _ = np.histogram2d(distances[1:], angles[1:], bins=[r_bins, theta_bins])
                contexts.append(hist.flatten())
                
            return np.array(contexts)
        
        # Compute shape contexts
        context1 = compute_shape_context(embeddings1, n_bins_r, n_bins_theta)
        context2 = compute_shape_context(embeddings2, n_bins_r, n_bins_theta)
        
        # Find best matching using Hungarian algorithm
        from scipy.optimize import linear_sum_assignment
        from scipy.spatial.distance import cdist
        
        # Compute chi-square distance between contexts
        cost_matrix = cdist(context1, context2, metric='chi2')
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        return float(np.mean(cost_matrix[row_indices, col_indices]))
        
    def persistent_homology_distance(self, 
                                    embeddings1: np.ndarray, 
                                    embeddings2: np.ndarray) -> Dict[str, float]:
        """Compute distances based on persistent homology (simplified version).
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            
        Returns:
            Dictionary of topological distance measures
        """
        try:
            # Simplified topological analysis using distance-based features
            
            # Compute persistence-like features
            def compute_persistence_features(embeddings, max_dimension=1):
                """Compute simplified persistence features."""
                from scipy.spatial.distance import pdist
                
                distances = pdist(embeddings)
                
                # Estimate birth and death times for connected components
                sorted_distances = np.sort(distances)
                
                features = {
                    'birth_time_mean': float(np.mean(sorted_distances[:len(sorted_distances)//10])),
                    'death_time_mean': float(np.mean(sorted_distances[-len(sorted_distances)//10:])),
                    'persistence_mean': float(np.mean(sorted_distances)),
                    'persistence_std': float(np.std(sorted_distances)),
                    'max_persistence': float(np.max(sorted_distances)),
                    'n_components_estimate': len(embeddings)  # Simplified estimate
                }
                
                return features
            
            features1 = compute_persistence_features(embeddings1)
            features2 = compute_persistence_features(embeddings2)
            
            # Compute differences
            persistence_distance = {
                'birth_time_diff': abs(features1['birth_time_mean'] - features2['birth_time_mean']),
                'death_time_diff': abs(features1['death_time_mean'] - features2['death_time_mean']),
                'persistence_mean_diff': abs(features1['persistence_mean'] - features2['persistence_mean']),
                'persistence_std_diff': abs(features1['persistence_std'] - features2['persistence_std']),
                'max_persistence_diff': abs(features1['max_persistence'] - features2['max_persistence']),
                'bottleneck_approximation': max(
                    abs(features1['birth_time_mean'] - features2['birth_time_mean']),
                    abs(features1['death_time_mean'] - features2['death_time_mean'])
                )
            }
            
            return {k: float(v) for k, v in persistence_distance.items()}
            
        except Exception as e:
            self.logger.warning(f"Persistent homology computation failed: {e}")
            return {'bottleneck_approximation': float('inf')}
            
    def manifold_distance(self, 
                         embeddings1: np.ndarray, 
                         embeddings2: np.ndarray,
                         method: str = 'procrustes') -> float:
        """Compute distance between embedding manifolds.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            method: Method for manifold comparison ('procrustes', 'mds')
            
        Returns:
            Manifold distance
        """
        if method == 'procrustes':
            from scipy.linalg import orthogonal_procrustes
            
            # Align embeddings using Procrustes analysis
            R, scale = orthogonal_procrustes(embeddings1, embeddings2)
            aligned_embeddings1 = embeddings1 @ R
            
            # Compute residual after alignment
            residual = np.linalg.norm(aligned_embeddings1 - embeddings2, 'fro')
            return float(residual / np.sqrt(embeddings1.shape[0] * embeddings1.shape[1]))
            
        elif method == 'mds':
            # Use MDS to compare intrinsic geometry
            def mds_embedding(embeddings, n_components=2):
                distances = squareform(pdist(embeddings))
                mds = MDS(n_components=n_components, dissimilarity='precomputed', random_state=42)
                return mds.fit_transform(distances)
                
            try:
                mds1 = mds_embedding(embeddings1)
                mds2 = mds_embedding(embeddings2)
                
                # Align MDS embeddings
                from scipy.linalg import orthogonal_procrustes
                R, _ = orthogonal_procrustes(mds1, mds2)
                aligned_mds1 = mds1 @ R
                
                return float(np.linalg.norm(aligned_mds1 - mds2, 'fro'))
                
            except Exception as e:
                self.logger.warning(f"MDS computation failed: {e}")
                return float('inf')
                
        else:
            raise ValueError(f"Unknown manifold distance method: {method}")
            
    def compute_comprehensive_distances(self, 
                                       embeddings1: np.ndarray, 
                                       embeddings2: np.ndarray,
                                       sample_size: Optional[int] = 100) -> Dict[str, Union[float, Dict[str, float]]]:
        """Compute comprehensive geometric distance measures.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            sample_size: Sample size for expensive computations
            
        Returns:
            Dictionary of all computed distances
        """
        distances = {}
        
        # Sample if datasets are too large
        if sample_size is not None and (embeddings1.shape[0] > sample_size or embeddings2.shape[0] > sample_size):
            if embeddings1.shape[0] > sample_size:
                idx1 = np.random.choice(embeddings1.shape[0], sample_size, replace=False)
                emb1_sample = embeddings1[idx1]
            else:
                emb1_sample = embeddings1
                
            if embeddings2.shape[0] > sample_size:
                idx2 = np.random.choice(embeddings2.shape[0], sample_size, replace=False)
                emb2_sample = embeddings2[idx2]
            else:
                emb2_sample = embeddings2
        else:
            emb1_sample, emb2_sample = embeddings1, embeddings2
            
        # Basic geometric distances
        distances['hausdorff_distance'] = self.hausdorff_distance(emb1_sample, emb2_sample)
        distances['chamfer_distance'] = self.chamfer_distance(emb1_sample, emb2_sample)
        
        # More complex distances (with error handling)
        try:
            distances['gromov_hausdorff_approx'] = self.gromov_hausdorff_approximation(
                emb1_sample, emb2_sample, sample_size
            )
        except Exception as e:
            self.logger.warning(f"Gromov-Hausdorff approximation failed: {e}")
            distances['gromov_hausdorff_approx'] = float('inf')
            
        try:
            distances['shape_context_distance'] = self.shape_context_distance(emb1_sample, emb2_sample)
        except Exception as e:
            self.logger.warning(f"Shape context distance failed: {e}")
            distances['shape_context_distance'] = float('inf')
            
        # Topological distances
        distances['persistent_homology'] = self.persistent_homology_distance(emb1_sample, emb2_sample)
        
        # Manifold distances
        try:
            distances['procrustes_manifold_distance'] = self.manifold_distance(
                emb1_sample, emb2_sample, 'procrustes'
            )
        except Exception as e:
            self.logger.warning(f"Procrustes manifold distance failed: {e}")
            distances['procrustes_manifold_distance'] = float('inf')
            
        try:
            distances['mds_manifold_distance'] = self.manifold_distance(
                emb1_sample, emb2_sample, 'mds'
            )
        except Exception as e:
            self.logger.warning(f"MDS manifold distance failed: {e}")
            distances['mds_manifold_distance'] = float('inf')
            
        return distances
        
    def compute_distance_matrix(self, 
                               embeddings_dict: Dict[str, np.ndarray],
                               metric: str = 'hausdorff',
                               sample_size: Optional[int] = 100) -> Dict[str, Dict[str, float]]:
        """Compute distance matrix between all pairs of embedding sets.
        
        Args:
            embeddings_dict: Dictionary mapping language names to embeddings
            metric: Which geometric distance metric to use
            sample_size: Sample size for expensive computations
            
        Returns:
            Dictionary of dictionaries with pairwise distance values
        """
        languages = list(embeddings_dict.keys())
        distance_matrix = {}
        
        for i, lang1 in enumerate(languages):
            distance_matrix[lang1] = {}
            for j, lang2 in enumerate(languages):
                if i == j:
                    distance_matrix[lang1][lang2] = 0.0
                elif lang2 in distance_matrix and lang1 in distance_matrix[lang2]:
                    # Use symmetry
                    distance_matrix[lang1][lang2] = distance_matrix[lang2][lang1]
                else:
                    try:
                        if metric == 'hausdorff':
                            distance = self.hausdorff_distance(
                                embeddings_dict[lang1], embeddings_dict[lang2]
                            )
                        elif metric == 'chamfer':
                            distance = self.chamfer_distance(
                                embeddings_dict[lang1], embeddings_dict[lang2]
                            )
                        elif metric == 'gromov_hausdorff':
                            distance = self.gromov_hausdorff_approximation(
                                embeddings_dict[lang1], embeddings_dict[lang2], sample_size
                            )
                        else:
                            raise ValueError(f"Unknown metric: {metric}")
                            
                        distance_matrix[lang1][lang2] = distance
                        self.logger.info(f"Geometric {metric} {lang1}-{lang2}: {distance:.6f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to compute {metric} for {lang1}-{lang2}: {e}")
                        distance_matrix[lang1][lang2] = float('inf')
                        
        return distance_matrix