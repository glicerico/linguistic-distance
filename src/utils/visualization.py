"""Visualization utilities for linguistic distance analysis."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import logging
from sklearn.manifold import MDS, TSNE
from sklearn.decomposition import PCA


class VisualizationUtils:
    """Utilities for creating visualizations of linguistic distance data."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: Tuple[int, int] = (10, 8)):
        """Initialize visualization utilities.
        
        Args:
            style: Matplotlib style to use
            figsize: Default figure size
        """
        plt.style.use(style)
        self.default_figsize = figsize
        self.logger = logging.getLogger(__name__)
        
        # Set up color palette
        self.colors = sns.color_palette("husl", 10)
        
    def plot_distance_matrix(self, 
                            distance_matrix: Dict[str, Dict[str, float]],
                            title: str = "Linguistic Distance Matrix",
                            save_path: Optional[str] = None,
                            annotate: bool = True,
                            colormap: str = 'viridis') -> plt.Figure:
        """Plot a heatmap of linguistic distances.
        
        Args:
            distance_matrix: Dictionary of dictionaries with distances
            title: Title for the plot
            save_path: Path to save the figure
            annotate: Whether to annotate cells with values
            colormap: Colormap to use
            
        Returns:
            matplotlib Figure object
        """
        # Convert to DataFrame
        languages = list(distance_matrix.keys())
        matrix_data = []
        
        for lang1 in languages:
            row = []
            for lang2 in languages:
                value = distance_matrix[lang1].get(lang2, np.nan)
                # Handle infinite values
                if np.isinf(value):
                    value = np.nan
                row.append(value)
            matrix_data.append(row)
            
        df = pd.DataFrame(matrix_data, index=languages, columns=languages)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Create heatmap
        mask = df.isna()
        sns.heatmap(df, annot=annotate, cmap=colormap, mask=mask, 
                   square=True, ax=ax, fmt='.3f',
                   cbar_kws={'label': 'Distance'})
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Language', fontsize=12)
        ax.set_ylabel('Language', fontsize=12)
        
        # Rotate labels for better readability
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved distance matrix plot to {save_path}")
            
        return fig
        
    def plot_embeddings_2d(self, 
                           embeddings_dict: Dict[str, np.ndarray],
                           method: str = 'pca',
                           title: str = "Embedding Visualization",
                           save_path: Optional[str] = None,
                           n_samples_per_lang: int = 100) -> plt.Figure:
        """Plot 2D visualization of embeddings from multiple languages.
        
        Args:
            embeddings_dict: Dictionary mapping language names to embeddings
            method: Dimensionality reduction method ('pca', 'tsne', 'mds')
            title: Title for the plot
            save_path: Path to save the figure
            n_samples_per_lang: Number of samples per language to plot
            
        Returns:
            matplotlib Figure object
        """
        # Sample embeddings for visualization
        all_embeddings = []
        all_labels = []
        colors_list = []
        
        color_map = {lang: self.colors[i % len(self.colors)] 
                    for i, lang in enumerate(embeddings_dict.keys())}
        
        for lang, embeddings in embeddings_dict.items():
            n_samples = min(n_samples_per_lang, embeddings.shape[0])
            if n_samples > 0:
                indices = np.random.choice(embeddings.shape[0], n_samples, replace=False)
                sampled_embeddings = embeddings[indices]
                
                all_embeddings.append(sampled_embeddings)
                all_labels.extend([lang] * n_samples)
                colors_list.extend([color_map[lang]] * n_samples)
                
        if not all_embeddings:
            raise ValueError("No embeddings to plot")
            
        # Combine all embeddings
        combined_embeddings = np.vstack(all_embeddings)
        
        # Dimensionality reduction
        if method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(combined_embeddings)
            explained_variance = reducer.explained_variance_ratio_
            subtitle = f"Explained variance: {explained_variance[0]:.1%}, {explained_variance[1]:.1%}"
        elif method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, combined_embeddings.shape[0]-1))
            reduced_embeddings = reducer.fit_transform(combined_embeddings)
            subtitle = f"Perplexity: {reducer.perplexity}"
        elif method == 'mds':
            reducer = MDS(n_components=2, random_state=42)
            reduced_embeddings = reducer.fit_transform(combined_embeddings)
            subtitle = f"Stress: {reducer.stress_:.3f}"
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")
            
        # Create plot
        fig, ax = plt.subplots(figsize=self.default_figsize)
        
        # Plot each language separately for proper legend
        for lang, color in color_map.items():
            lang_mask = np.array(all_labels) == lang
            if np.any(lang_mask):
                lang_points = reduced_embeddings[lang_mask]
                ax.scatter(lang_points[:, 0], lang_points[:, 1], 
                          c=[color], label=lang, alpha=0.7, s=20)
                
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'{title}\n{subtitle}', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved embedding plot to {save_path}")
            
        return fig
        
    def plot_embedding_statistics(self, 
                                 stats_dict: Dict[str, Dict[str, float]],
                                 metrics: List[str] = None,
                                 title: str = "Embedding Statistics",
                                 save_path: Optional[str] = None) -> plt.Figure:
        """Plot embedding statistics across languages.
        
        Args:
            stats_dict: Dictionary mapping languages to statistics
            metrics: List of metrics to plot (all if None)
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        # Convert to DataFrame
        df = pd.DataFrame(stats_dict).T
        
        if metrics is None:
            metrics = [col for col in df.columns if isinstance(df[col].iloc[0], (int, float))]
            
        # Filter metrics
        df_filtered = df[metrics]
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
            
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            # Bar plot
            languages = df_filtered.index
            values = df_filtered[metric]
            
            bars = ax.bar(languages, values, color=self.colors[:len(languages)])
            ax.set_title(metric.replace('_', ' ').title())
            ax.set_ylabel('Value')
            
            # Rotate x-axis labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{value:.3f}', ha='center', va='bottom')
                       
        # Hide unused subplots
        for i in range(n_metrics, len(axes)):
            axes[i].set_visible(False)
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved statistics plot to {save_path}")
            
        return fig
        
    def plot_distance_comparison(self, 
                                distance_matrices: Dict[str, Dict[str, Dict[str, float]]],
                                title: str = "Distance Method Comparison",
                                save_path: Optional[str] = None) -> plt.Figure:
        """Compare different distance metrics side by side.
        
        Args:
            distance_matrices: Dictionary mapping method names to distance matrices
            title: Title for the plot
            save_path: Path to save the figure
            
        Returns:
            matplotlib Figure object
        """
        n_methods = len(distance_matrices)
        n_cols = min(3, n_methods)
        n_rows = (n_methods + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
        if n_methods == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
            
        for i, (method_name, distance_matrix) in enumerate(distance_matrices.items()):
            ax = axes[i]
            
            # Convert to DataFrame
            languages = list(distance_matrix.keys())
            matrix_data = []
            
            for lang1 in languages:
                row = []
                for lang2 in languages:
                    value = distance_matrix[lang1].get(lang2, np.nan)
                    if np.isinf(value):
                        value = np.nan
                    row.append(value)
                matrix_data.append(row)
                
            df = pd.DataFrame(matrix_data, index=languages, columns=languages)
            
            # Create heatmap
            mask = df.isna()
            sns.heatmap(df, annot=True, cmap='viridis', mask=mask, 
                       square=True, ax=ax, fmt='.3f',
                       cbar=True)
            
            ax.set_title(method_name.replace('_', ' ').title())
            
            # Rotate labels
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            plt.setp(ax.get_yticklabels(), rotation=0)
            
        # Hide unused subplots
        for i in range(n_methods, len(axes)):
            axes[i].set_visible(False)
            
        fig.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved comparison plot to {save_path}")
            
        return fig
        
    def plot_dendrogram(self, 
                       distance_matrix: Dict[str, Dict[str, float]],
                       title: str = "Linguistic Distance Dendrogram",
                       save_path: Optional[str] = None,
                       method: str = 'ward') -> plt.Figure:
        """Create a dendrogram from distance matrix.
        
        Args:
            distance_matrix: Dictionary of dictionaries with distances
            title: Title for the plot
            save_path: Path to save the figure
            method: Linkage method for clustering
            
        Returns:
            matplotlib Figure object
        """
        from scipy.cluster.hierarchy import dendrogram, linkage
        from scipy.spatial.distance import squareform
        
        # Convert to condensed distance matrix
        languages = list(distance_matrix.keys())
        n_langs = len(languages)
        
        # Create symmetric matrix
        matrix = np.zeros((n_langs, n_langs))
        for i, lang1 in enumerate(languages):
            for j, lang2 in enumerate(languages):
                if i != j:
                    value = distance_matrix[lang1].get(lang2, np.inf)
                    if np.isinf(value) or np.isnan(value):
                        # Use maximum finite value + small increment
                        finite_values = []
                        for l1 in distance_matrix:
                            for l2, v in distance_matrix[l1].items():
                                if l1 != l2 and np.isfinite(v):
                                    finite_values.append(v)
                        if finite_values:
                            value = max(finite_values) * 1.1
                        else:
                            value = 1.0
                    matrix[i, j] = value
                    
        # Convert to condensed form
        condensed_matrix = squareform(matrix)
        
        # Perform hierarchical clustering
        linkage_matrix = linkage(condensed_matrix, method=method)
        
        # Create dendrogram
        fig, ax = plt.subplots(figsize=(12, 8))
        
        dendrogram(linkage_matrix, labels=languages, ax=ax, 
                  orientation='top', distance_sort='descending')
        
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.set_xlabel('Languages', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Saved dendrogram to {save_path}")
            
        return fig


# Convenience functions
def plot_distance_matrix(distance_matrix: Dict[str, Dict[str, float]],
                        title: str = "Linguistic Distance Matrix",
                        save_path: Optional[str] = None,
                        **kwargs) -> plt.Figure:
    """Convenience function to plot distance matrix."""
    viz = VisualizationUtils()
    return viz.plot_distance_matrix(distance_matrix, title, save_path, **kwargs)


def plot_embeddings(embeddings_dict: Dict[str, np.ndarray],
                   method: str = 'pca',
                   title: str = "Embedding Visualization",
                   save_path: Optional[str] = None,
                   **kwargs) -> plt.Figure:
    """Convenience function to plot embeddings."""
    viz = VisualizationUtils()
    return viz.plot_embeddings_2d(embeddings_dict, method, title, save_path, **kwargs)