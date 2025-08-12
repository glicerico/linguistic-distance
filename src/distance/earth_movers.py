"""Earth Mover's Distance (Wasserstein Distance) for embedding spaces."""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import warnings
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
import logging

# Try to import optimal transport library
try:
    import ot
    HAS_POT = True
except ImportError:
    HAS_POT = False
    warnings.warn("POT (Python Optimal Transport) not available. Using approximation methods.")


class EarthMoversDistance:
    """Compute Earth Mover's Distance between embedding spaces."""
    
    def __init__(self, method: str = "sinkhorn", reg: float = 0.1):
        """Initialize Earth Mover's Distance calculator.
        
        Args:
            method: Method to use ('exact', 'sinkhorn', 'approximation')
            reg: Regularization parameter for Sinkhorn algorithm
        """
        self.method = method
        self.reg = reg
        self.logger = logging.getLogger(__name__)
        
        if method in ['exact', 'sinkhorn'] and not HAS_POT:
            self.logger.warning(f"POT not available, falling back to approximation method")
            self.method = 'approximation'
            
    def compute_distance_matrix(self, 
                              embeddings1: np.ndarray, 
                              embeddings2: np.ndarray,
                              metric: str = 'euclidean') -> np.ndarray:
        """Compute pairwise distance matrix between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings (n1 x d)
            embeddings2: Second set of embeddings (n2 x d)
            metric: Distance metric to use
            
        Returns:
            Distance matrix (n1 x n2)
        """
        return cdist(embeddings1, embeddings2, metric=metric)
        
    def emd_exact(self, 
                  embeddings1: np.ndarray, 
                  embeddings2: np.ndarray,
                  weights1: Optional[np.ndarray] = None,
                  weights2: Optional[np.ndarray] = None) -> float:
        """Compute exact Earth Mover's Distance using linear programming.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution (uniform if None)
            weights2: Weights for second distribution (uniform if None)
            
        Returns:
            Earth Mover's Distance
        """
        if not HAS_POT:
            raise ImportError("POT library required for exact EMD computation")
            
        n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
        
        if weights1 is None:
            weights1 = np.ones(n1) / n1
        if weights2 is None:
            weights2 = np.ones(n2) / n2
            
        # Normalize weights
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)
        
        # Compute distance matrix
        cost_matrix = self.compute_distance_matrix(embeddings1, embeddings2)
        
        # Compute EMD
        emd_value = ot.emd2(weights1, weights2, cost_matrix)
        
        return float(emd_value)
        
    def emd_sinkhorn(self, 
                     embeddings1: np.ndarray, 
                     embeddings2: np.ndarray,
                     weights1: Optional[np.ndarray] = None,
                     weights2: Optional[np.ndarray] = None,
                     reg: Optional[float] = None) -> float:
        """Compute Earth Mover's Distance using Sinkhorn regularization.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution
            weights2: Weights for second distribution
            reg: Regularization parameter (uses self.reg if None)
            
        Returns:
            Regularized Earth Mover's Distance
        """
        if not HAS_POT:
            raise ImportError("POT library required for Sinkhorn EMD computation")
            
        if reg is None:
            reg = self.reg
            
        n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
        
        if weights1 is None:
            weights1 = np.ones(n1) / n1
        if weights2 is None:
            weights2 = np.ones(n2) / n2
            
        # Normalize weights
        weights1 = weights1 / np.sum(weights1)
        weights2 = weights2 / np.sum(weights2)
        
        # Compute distance matrix
        cost_matrix = self.compute_distance_matrix(embeddings1, embeddings2)
        
        # Compute regularized EMD
        emd_value = ot.sinkhorn2(weights1, weights2, cost_matrix, reg)
        
        return float(emd_value)
        
    def emd_approximation(self, 
                         embeddings1: np.ndarray, 
                         embeddings2: np.ndarray,
                         weights1: Optional[np.ndarray] = None,
                         weights2: Optional[np.ndarray] = None,
                         method: str = 'hungarian') -> float:
        """Compute approximation of Earth Mover's Distance.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution
            weights2: Weights for second distribution
            method: Approximation method ('hungarian', 'centroid')
            
        Returns:
            Approximated Earth Mover's Distance
        """
        n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
        
        if weights1 is None:
            weights1 = np.ones(n1) / n1
        if weights2 is None:
            weights2 = np.ones(n2) / n2
            
        if method == 'hungarian':
            return self._hungarian_approximation(embeddings1, embeddings2, weights1, weights2)
        elif method == 'centroid':
            return self._centroid_distance(embeddings1, embeddings2, weights1, weights2)
        else:
            raise ValueError(f"Unknown approximation method: {method}")
            
    def _hungarian_approximation(self, 
                                embeddings1: np.ndarray, 
                                embeddings2: np.ndarray,
                                weights1: np.ndarray, 
                                weights2: np.ndarray) -> float:
        """Approximate EMD using Hungarian algorithm for bipartite matching.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution
            weights2: Weights for second distribution
            
        Returns:
            Hungarian approximation of EMD
        """
        # Make the problem square by padding if necessary
        n1, n2 = embeddings1.shape[0], embeddings2.shape[0]
        
        if n1 != n2:
            if n1 < n2:
                # Pad embeddings1
                padding = np.zeros((n2 - n1, embeddings1.shape[1]))
                embeddings1 = np.vstack([embeddings1, padding])
                weights1 = np.concatenate([weights1, np.zeros(n2 - n1)])
            else:
                # Pad embeddings2
                padding = np.zeros((n1 - n2, embeddings2.shape[1]))
                embeddings2 = np.vstack([embeddings2, padding])
                weights2 = np.concatenate([weights2, np.zeros(n1 - n2)])
                
        # Compute cost matrix
        cost_matrix = self.compute_distance_matrix(embeddings1, embeddings2)
        
        # Solve assignment problem
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        
        # Compute total cost weighted by distribution weights
        total_cost = 0.0
        for i, j in zip(row_indices, col_indices):
            weight = min(weights1[i], weights2[j])
            total_cost += cost_matrix[i, j] * weight
            
        return float(total_cost)
        
    def _centroid_distance(self, 
                          embeddings1: np.ndarray, 
                          embeddings2: np.ndarray,
                          weights1: np.ndarray, 
                          weights2: np.ndarray) -> float:
        """Compute distance between weighted centroids as EMD approximation.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution
            weights2: Weights for second distribution
            
        Returns:
            Distance between centroids
        """
        # Compute weighted centroids
        centroid1 = np.average(embeddings1, axis=0, weights=weights1)
        centroid2 = np.average(embeddings2, axis=0, weights=weights2)
        
        # Return Euclidean distance between centroids
        return float(np.linalg.norm(centroid1 - centroid2))
        
    def compute_emd(self, 
                   embeddings1: np.ndarray, 
                   embeddings2: np.ndarray,
                   weights1: Optional[np.ndarray] = None,
                   weights2: Optional[np.ndarray] = None,
                   method: Optional[str] = None) -> float:
        """Compute Earth Mover's Distance between two embedding sets.
        
        Args:
            embeddings1: First embedding matrix
            embeddings2: Second embedding matrix
            weights1: Weights for first distribution
            weights2: Weights for second distribution
            method: Method to use (overrides self.method)
            
        Returns:
            Earth Mover's Distance
        """
        if method is None:
            method = self.method
            
        if method == 'exact':
            return self.emd_exact(embeddings1, embeddings2, weights1, weights2)
        elif method == 'sinkhorn':
            return self.emd_sinkhorn(embeddings1, embeddings2, weights1, weights2)
        elif method == 'approximation':
            return self.emd_approximation(embeddings1, embeddings2, weights1, weights2)
        else:
            raise ValueError(f"Unknown EMD method: {method}")
            
    def compute_emd_matrix(self, 
                          embeddings_dict: Dict[str, np.ndarray],
                          weights_dict: Optional[Dict[str, np.ndarray]] = None,
                          method: Optional[str] = None) -> Dict[str, Dict[str, float]]:
        """Compute EMD matrix between all pairs of embedding sets.
        
        Args:
            embeddings_dict: Dictionary mapping language names to embeddings
            weights_dict: Dictionary mapping language names to weights
            method: Method to use
            
        Returns:
            Dictionary of dictionaries with pairwise EMD values
        """
        languages = list(embeddings_dict.keys())
        emd_matrix = {}
        
        if weights_dict is None:
            weights_dict = {}
            
        for i, lang1 in enumerate(languages):
            emd_matrix[lang1] = {}
            for j, lang2 in enumerate(languages):
                if i == j:
                    emd_matrix[lang1][lang2] = 0.0
                elif lang2 in emd_matrix and lang1 in emd_matrix[lang2]:
                    # Use symmetry
                    emd_matrix[lang1][lang2] = emd_matrix[lang2][lang1]
                else:
                    weights1 = weights_dict.get(lang1, None)
                    weights2 = weights_dict.get(lang2, None)
                    
                    try:
                        distance = self.compute_emd(
                            embeddings_dict[lang1], 
                            embeddings_dict[lang2],
                            weights1, 
                            weights2, 
                            method
                        )
                        emd_matrix[lang1][lang2] = distance
                        
                        self.logger.info(f"EMD {lang1}-{lang2}: {distance:.6f}")
                        
                    except Exception as e:
                        self.logger.error(f"Failed to compute EMD for {lang1}-{lang2}: {e}")
                        emd_matrix[lang1][lang2] = float('inf')
                        
        return emd_matrix
        
    def compute_distribution_statistics(self, 
                                      embeddings: np.ndarray,
                                      weights: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Compute statistics of an embedding distribution.
        
        Args:
            embeddings: Embedding matrix
            weights: Optional weights
            
        Returns:
            Dictionary of distribution statistics
        """
        if weights is None:
            weights = np.ones(embeddings.shape[0]) / embeddings.shape[0]
        else:
            weights = weights / np.sum(weights)
            
        # Weighted statistics
        mean = np.average(embeddings, axis=0, weights=weights)
        variance = np.average((embeddings - mean) ** 2, axis=0, weights=weights)
        
        # Pairwise distances within the distribution
        distances = cdist(embeddings, embeddings)
        
        # Weighted average intra-distribution distance
        weighted_distances = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                weighted_distances.append(distances[i, j] * weights[i] * weights[j])
                
        return {
            'n_points': embeddings.shape[0],
            'dimension': embeddings.shape[1],
            'mean_norm': float(np.linalg.norm(mean)),
            'mean_variance': float(np.mean(variance)),
            'total_variance': float(np.sum(variance)),
            'diameter': float(np.max(distances)),
            'mean_intra_distance': float(np.mean(weighted_distances)) if weighted_distances else 0.0,
            'effective_dimension': float(np.sum(variance > 1e-10))  # Dimensions with non-zero variance
        }