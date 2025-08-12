"""Linear mapping methods for embedding space alignment."""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from sklearn.linear_model import Ridge, LinearRegression
from scipy.linalg import orthogonal_procrustes
import logging


class LinearMapping:
    """Linear mapping methods for aligning embedding spaces."""
    
    def __init__(self):
        """Initialize the linear mapping class."""
        self.logger = logging.getLogger(__name__)
        self.transformation_matrix = None
        self.translation_vector = None
        self.method = None
        
    def fit_linear_regression(self, 
                             source_embeddings: np.ndarray, 
                             target_embeddings: np.ndarray,
                             regularization: float = 0.0) -> 'LinearMapping':
        """Fit a linear regression mapping from source to target embeddings.
        
        Args:
            source_embeddings: Source embedding matrix (n_words x n_dim)
            target_embeddings: Target embedding matrix (n_words x n_dim)
            regularization: L2 regularization strength
            
        Returns:
            Self for method chaining
        """
        if source_embeddings.shape != target_embeddings.shape:
            raise ValueError(f"Shape mismatch: {source_embeddings.shape} vs {target_embeddings.shape}")
            
        if regularization > 0:
            model = Ridge(alpha=regularization, fit_intercept=True)
        else:
            model = LinearRegression(fit_intercept=True)
            
        # Fit the model
        model.fit(source_embeddings, target_embeddings)
        
        # Store the learned parameters
        self.transformation_matrix = model.coef_.T
        self.translation_vector = model.intercept_
        self.method = f"linear_regression_reg_{regularization}"
        
        # Calculate fit quality
        predicted = model.predict(source_embeddings)
        mse = np.mean((predicted - target_embeddings) ** 2)
        
        self.logger.info(f"Linear regression mapping fitted with MSE: {mse:.6f}")
        
        return self
        
    def fit_orthogonal_mapping(self, 
                              source_embeddings: np.ndarray, 
                              target_embeddings: np.ndarray,
                              center: bool = True) -> 'LinearMapping':
        """Fit an orthogonal mapping using Procrustes analysis.
        
        Args:
            source_embeddings: Source embedding matrix
            target_embeddings: Target embedding matrix
            center: Whether to center the embeddings before alignment
            
        Returns:
            Self for method chaining
        """
        if source_embeddings.shape != target_embeddings.shape:
            raise ValueError(f"Shape mismatch: {source_embeddings.shape} vs {target_embeddings.shape}")
            
        source = source_embeddings.copy()
        target = target_embeddings.copy()
        
        # Center the embeddings if requested
        if center:
            source_mean = np.mean(source, axis=0)
            target_mean = np.mean(target, axis=0)
            source -= source_mean
            target -= target_mean
            self.translation_vector = target_mean - source_mean
        else:
            self.translation_vector = np.zeros(source.shape[1])
            
        # Compute the optimal orthogonal transformation
        R, scale = orthogonal_procrustes(source, target)
        
        self.transformation_matrix = R
        self.method = f"orthogonal_procrustes_center_{center}"
        
        # Calculate alignment error
        aligned = source @ R
        if center:
            aligned += self.translation_vector
        mse = np.mean((aligned - target_embeddings) ** 2)
        
        self.logger.info(f"Orthogonal mapping fitted with MSE: {mse:.6f}")
        
        return self
        
    def fit_canonical_correlation(self, 
                                 source_embeddings: np.ndarray, 
                                 target_embeddings: np.ndarray,
                                 n_components: Optional[int] = None,
                                 regularization: float = 1e-4) -> 'LinearMapping':
        """Fit a canonical correlation analysis mapping.
        
        Args:
            source_embeddings: Source embedding matrix
            target_embeddings: Target embedding matrix
            n_components: Number of canonical components (None for all)
            regularization: Regularization parameter
            
        Returns:
            Self for method chaining
        """
        if source_embeddings.shape != target_embeddings.shape:
            raise ValueError(f"Shape mismatch: {source_embeddings.shape} vs {target_embeddings.shape}")
            
        # Center the data
        X = source_embeddings - np.mean(source_embeddings, axis=0)
        Y = target_embeddings - np.mean(target_embeddings, axis=0)
        
        n_samples, n_features = X.shape
        
        if n_components is None:
            n_components = min(n_samples, n_features)
            
        # Compute covariance matrices
        Cxx = (X.T @ X) / (n_samples - 1) + regularization * np.eye(n_features)
        Cyy = (Y.T @ Y) / (n_samples - 1) + regularization * np.eye(n_features)
        Cxy = (X.T @ Y) / (n_samples - 1)
        
        # Solve generalized eigenvalue problem
        try:
            # Cxx^{-1/2} Cxy Cyy^{-1} Cyx Cxx^{-1/2} = U Lambda U^T
            Cxx_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cxx))
            Cyy_inv_sqrt = np.linalg.inv(np.linalg.cholesky(Cyy))
            
            M = Cxx_inv_sqrt.T @ Cxy @ np.linalg.inv(Cyy) @ Cxy.T @ Cxx_inv_sqrt
            eigenvals, eigenvecs = np.linalg.eigh(M)
            
            # Sort by eigenvalue (correlation)
            idx = np.argsort(eigenvals)[::-1]
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
            
            # Get canonical vectors
            U = Cxx_inv_sqrt @ eigenvecs[:, :n_components]
            V = Cyy_inv_sqrt @ np.linalg.inv(Cyy) @ Cxy.T @ Cxx_inv_sqrt @ eigenvecs[:, :n_components]
            
            # The transformation matrix
            self.transformation_matrix = U @ V.T
            self.translation_vector = np.mean(target_embeddings, axis=0) - np.mean(source_embeddings, axis=0)
            self.method = f"canonical_correlation_{n_components}_components"
            
            # Calculate correlation
            max_corr = np.sqrt(eigenvals[0])
            self.logger.info(f"CCA mapping fitted with max correlation: {max_corr:.4f}")
            
        except np.linalg.LinAlgError as e:
            self.logger.warning(f"CCA failed, falling back to linear regression: {e}")
            return self.fit_linear_regression(source_embeddings, target_embeddings, regularization)
            
        return self
        
    def transform(self, source_embeddings: np.ndarray) -> np.ndarray:
        """Apply the learned transformation to source embeddings.
        
        Args:
            source_embeddings: Source embeddings to transform
            
        Returns:
            Transformed embeddings
        """
        if self.transformation_matrix is None:
            raise ValueError("No transformation has been fitted yet")
            
        transformed = source_embeddings @ self.transformation_matrix
        
        if self.translation_vector is not None:
            transformed += self.translation_vector
            
        return transformed
        
    def fit_transform(self, 
                     source_embeddings: np.ndarray, 
                     target_embeddings: np.ndarray,
                     method: str = "orthogonal",
                     **kwargs) -> np.ndarray:
        """Fit transformation and apply it in one step.
        
        Args:
            source_embeddings: Source embedding matrix
            target_embeddings: Target embedding matrix
            method: Alignment method ('linear', 'orthogonal', 'cca')
            **kwargs: Additional arguments for the method
            
        Returns:
            Transformed source embeddings
        """
        if method == "linear":
            self.fit_linear_regression(source_embeddings, target_embeddings, **kwargs)
        elif method == "orthogonal":
            self.fit_orthogonal_mapping(source_embeddings, target_embeddings, **kwargs)
        elif method == "cca":
            self.fit_canonical_correlation(source_embeddings, target_embeddings, **kwargs)
        else:
            raise ValueError(f"Unknown alignment method: {method}")
            
        return self.transform(source_embeddings)
        
    def evaluate_alignment(self, 
                          source_embeddings: np.ndarray, 
                          target_embeddings: np.ndarray) -> Dict[str, float]:
        """Evaluate the quality of the alignment.
        
        Args:
            source_embeddings: Source embeddings
            target_embeddings: Target embeddings (ground truth)
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.transformation_matrix is None:
            raise ValueError("No transformation has been fitted yet")
            
        aligned = self.transform(source_embeddings)
        
        # Mean squared error
        mse = np.mean((aligned - target_embeddings) ** 2)
        
        # Mean absolute error
        mae = np.mean(np.abs(aligned - target_embeddings))
        
        # Cosine similarity (average across words)
        cosine_similarities = []
        for i in range(len(aligned)):
            cos_sim = np.dot(aligned[i], target_embeddings[i]) / (
                np.linalg.norm(aligned[i]) * np.linalg.norm(target_embeddings[i])
            )
            cosine_similarities.append(cos_sim)
        mean_cosine_similarity = np.mean(cosine_similarities)
        
        # Correlation coefficient
        aligned_flat = aligned.flatten()
        target_flat = target_embeddings.flatten()
        correlation = np.corrcoef(aligned_flat, target_flat)[0, 1]
        
        return {
            'mse': float(mse),
            'mae': float(mae),
            'mean_cosine_similarity': float(mean_cosine_similarity),
            'correlation': float(correlation),
            'method': self.method
        }
        
    def get_transformation_info(self) -> Dict[str, Any]:
        """Get information about the learned transformation.
        
        Returns:
            Dictionary with transformation details
        """
        if self.transformation_matrix is None:
            return {'fitted': False}
            
        # Analyze the transformation matrix
        U, s, Vt = np.linalg.svd(self.transformation_matrix)
        
        return {
            'fitted': True,
            'method': self.method,
            'transformation_shape': self.transformation_matrix.shape,
            'has_translation': self.translation_vector is not None,
            'singular_values': s.tolist(),
            'condition_number': float(s[0] / s[-1]),
            'determinant': float(np.linalg.det(self.transformation_matrix)),
            'frobenius_norm': float(np.linalg.norm(self.transformation_matrix, 'fro'))
        }