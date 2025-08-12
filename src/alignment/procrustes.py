"""Procrustes alignment methods for embedding spaces."""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist, squareform
import logging


class ProcrustesAlignment:
    """Procrustes analysis for embedding space alignment."""
    
    def __init__(self):
        """Initialize the Procrustes alignment class."""
        self.logger = logging.getLogger(__name__)
        self.rotation_matrix = None
        self.translation_source = None
        self.translation_target = None
        self.scale_factor = None
        self.disparity = None
        
    def fit(self, 
            source_embeddings: np.ndarray, 
            target_embeddings: np.ndarray,
            scaling: bool = False,
            reflection: bool = True) -> 'ProcrustesAlignment':
        """Fit Procrustes transformation between embedding spaces.
        
        Args:
            source_embeddings: Source embedding matrix (n_words x n_dim)
            target_embeddings: Target embedding matrix (n_words x n_dim)
            scaling: Whether to allow uniform scaling
            reflection: Whether to allow reflection (determinant can be negative)
            
        Returns:
            Self for method chaining
        """
        if source_embeddings.shape != target_embeddings.shape:
            raise ValueError(f"Shape mismatch: {source_embeddings.shape} vs {target_embeddings.shape}")
            
        # Center both sets of embeddings
        source_centered = source_embeddings - np.mean(source_embeddings, axis=0)
        target_centered = target_embeddings - np.mean(target_embeddings, axis=0)
        
        # Store translation vectors
        self.translation_source = np.mean(source_embeddings, axis=0)
        self.translation_target = np.mean(target_embeddings, axis=0)
        
        # Compute the cross-covariance matrix
        H = source_centered.T @ target_centered
        
        # Singular value decomposition
        U, s, Vt = np.linalg.svd(H)
        
        # Compute rotation matrix
        R = Vt.T @ U.T
        
        # Handle reflection if not allowed
        if not reflection and np.linalg.det(R) < 0:
            # Flip the last column of Vt
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            s[-1] *= -1
            
        self.rotation_matrix = R
        
        # Compute scale factor if allowed
        if scaling:
            numerator = np.sum(s)
            denominator = np.sum(source_centered ** 2)
            self.scale_factor = numerator / denominator if denominator > 0 else 1.0
        else:
            self.scale_factor = 1.0
            
        # Compute disparity (goodness of fit)
        transformed_source = self.transform(source_embeddings)
        self.disparity = np.sum((transformed_source - target_embeddings) ** 2)
        
        self.logger.info(f"Procrustes alignment fitted:")
        self.logger.info(f"  Scale factor: {self.scale_factor:.6f}")
        self.logger.info(f"  Disparity: {self.disparity:.6f}")
        self.logger.info(f"  Rotation determinant: {np.linalg.det(self.rotation_matrix):.6f}")
        
        return self
        
    def transform(self, source_embeddings: np.ndarray) -> np.ndarray:
        """Apply Procrustes transformation to source embeddings.
        
        Args:
            source_embeddings: Embeddings to transform
            
        Returns:
            Transformed embeddings
        """
        if self.rotation_matrix is None:
            raise ValueError("Procrustes transformation not fitted yet")
            
        # Center, scale, rotate, and translate
        centered = source_embeddings - self.translation_source
        scaled = centered * self.scale_factor
        rotated = scaled @ self.rotation_matrix
        transformed = rotated + self.translation_target
        
        return transformed
        
    def fit_transform(self, 
                     source_embeddings: np.ndarray, 
                     target_embeddings: np.ndarray,
                     **kwargs) -> np.ndarray:
        """Fit and transform in one step.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            **kwargs: Arguments for fit method
            
        Returns:
            Transformed source embeddings
        """
        self.fit(source_embeddings, target_embeddings, **kwargs)
        return self.transform(source_embeddings)
        
    def inverse_transform(self, transformed_embeddings: np.ndarray) -> np.ndarray:
        """Apply inverse transformation.
        
        Args:
            transformed_embeddings: Transformed embeddings to revert
            
        Returns:
            Original space embeddings
        """
        if self.rotation_matrix is None:
            raise ValueError("Procrustes transformation not fitted yet")
            
        # Reverse the transformation steps
        translated = transformed_embeddings - self.translation_target
        rotated_back = translated @ self.rotation_matrix.T
        scaled_back = rotated_back / self.scale_factor
        original = scaled_back + self.translation_source
        
        return original
        
    def compute_distance_preservation(self, 
                                    source_embeddings: np.ndarray, 
                                    target_embeddings: np.ndarray) -> Dict[str, float]:
        """Compute how well pairwise distances are preserved.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            
        Returns:
            Dictionary with distance preservation metrics
        """
        if self.rotation_matrix is None:
            raise ValueError("Procrustes transformation not fitted yet")
            
        # Transform source embeddings
        transformed_source = self.transform(source_embeddings)
        
        # Compute pairwise distances
        source_distances = pdist(source_embeddings)
        target_distances = pdist(target_embeddings)
        transformed_distances = pdist(transformed_source)
        
        # Correlation between original source and target distances
        source_target_corr = np.corrcoef(source_distances, target_distances)[0, 1]
        
        # Correlation between transformed source and target distances
        transformed_target_corr = np.corrcoef(transformed_distances, target_distances)[0, 1]
        
        # RMS difference in distances
        source_target_rms = np.sqrt(np.mean((source_distances - target_distances) ** 2))
        transformed_target_rms = np.sqrt(np.mean((transformed_distances - target_distances) ** 2))
        
        return {
            'source_target_distance_correlation': float(source_target_corr),
            'transformed_target_distance_correlation': float(transformed_target_corr),
            'source_target_distance_rms': float(source_target_rms),
            'transformed_target_distance_rms': float(transformed_target_rms),
            'distance_improvement': float(source_target_rms - transformed_target_rms)
        }
        
    def evaluate_alignment(self, 
                          source_embeddings: np.ndarray, 
                          target_embeddings: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of Procrustes alignment.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.rotation_matrix is None:
            raise ValueError("Procrustes transformation not fitted yet")
            
        transformed = self.transform(source_embeddings)
        
        # Point-wise alignment metrics
        mse = np.mean((transformed - target_embeddings) ** 2)
        mae = np.mean(np.abs(transformed - target_embeddings))
        
        # Cosine similarity
        cosine_similarities = []
        for i in range(len(transformed)):
            cos_sim = np.dot(transformed[i], target_embeddings[i]) / (
                np.linalg.norm(transformed[i]) * np.linalg.norm(target_embeddings[i])
            )
            cosine_similarities.append(cos_sim)
        mean_cosine_similarity = np.mean(cosine_similarities)
        
        # Overall correlation
        correlation = np.corrcoef(transformed.flatten(), target_embeddings.flatten())[0, 1]
        
        # Distance preservation
        distance_metrics = self.compute_distance_preservation(source_embeddings, target_embeddings)
        
        # Combine all metrics
        metrics = {
            'mse': float(mse),
            'mae': float(mae),
            'mean_cosine_similarity': float(mean_cosine_similarity),
            'correlation': float(correlation),
            'disparity': float(self.disparity),
            'scale_factor': float(self.scale_factor),
            **distance_metrics
        }
        
        return metrics
        
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get detailed information about the Procrustes transformation.
        
        Returns:
            Dictionary with transformation details
        """
        if self.rotation_matrix is None:
            return {'fitted': False}
            
        # Analyze rotation matrix
        U, s, Vt = np.linalg.svd(self.rotation_matrix)
        
        # Check orthogonality
        should_be_identity = self.rotation_matrix.T @ self.rotation_matrix
        orthogonality_error = np.max(np.abs(should_be_identity - np.eye(self.rotation_matrix.shape[0])))
        
        return {
            'fitted': True,
            'rotation_matrix_shape': self.rotation_matrix.shape,
            'determinant': float(np.linalg.det(self.rotation_matrix)),
            'is_proper_rotation': np.linalg.det(self.rotation_matrix) > 0,
            'scale_factor': float(self.scale_factor),
            'disparity': float(self.disparity),
            'singular_values': s.tolist(),
            'condition_number': float(s[0] / s[-1]) if s[-1] > 0 else float('inf'),
            'orthogonality_error': float(orthogonality_error),
            'frobenius_norm': float(np.linalg.norm(self.rotation_matrix, 'fro')),
            'translation_norm_source': float(np.linalg.norm(self.translation_source)),
            'translation_norm_target': float(np.linalg.norm(self.translation_target))
        }
        
    def bootstrap_confidence(self, 
                            source_embeddings: np.ndarray, 
                            target_embeddings: np.ndarray,
                            n_bootstrap: int = 100,
                            confidence_level: float = 0.95) -> Dict[str, Tuple[float, float]]:
        """Compute bootstrap confidence intervals for alignment metrics.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            
        Returns:
            Dictionary mapping metric names to (lower, upper) confidence intervals
        """
        n_samples = source_embeddings.shape[0]
        
        bootstrap_metrics = {
            'mse': [],
            'correlation': [],
            'mean_cosine_similarity': []
        }
        
        for _ in range(n_bootstrap):
            # Bootstrap sample
            indices = np.random.choice(n_samples, n_samples, replace=True)
            source_sample = source_embeddings[indices]
            target_sample = target_embeddings[indices]
            
            # Fit and evaluate
            temp_alignment = ProcrustesAlignment()
            temp_alignment.fit(source_sample, target_sample)
            metrics = temp_alignment.evaluate_alignment(source_sample, target_sample)
            
            bootstrap_metrics['mse'].append(metrics['mse'])
            bootstrap_metrics['correlation'].append(metrics['correlation'])
            bootstrap_metrics['mean_cosine_similarity'].append(metrics['mean_cosine_similarity'])
            
        # Compute confidence intervals
        alpha = 1 - confidence_level
        confidence_intervals = {}
        
        for metric, values in bootstrap_metrics.items():
            values_sorted = np.sort(values)
            lower_idx = int(alpha / 2 * len(values))
            upper_idx = int((1 - alpha / 2) * len(values))
            
            confidence_intervals[metric] = (
                float(values_sorted[lower_idx]),
                float(values_sorted[upper_idx])
            )
            
        return confidence_intervals