"""Input/Output utilities for linguistic distance analysis."""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import logging
import csv
from datetime import datetime


class IOUtils:
    """Utilities for saving and loading linguistic distance data."""
    
    def __init__(self, base_dir: str = "results"):
        """Initialize I/O utilities.
        
        Args:
            base_dir: Base directory for saving results
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
    def save_embeddings(self, 
                       embeddings: Dict[str, np.ndarray],
                       filename: str,
                       format: str = 'pickle') -> None:
        """Save embeddings to disk.
        
        Args:
            embeddings: Dictionary mapping language names to embedding matrices
            filename: Filename to save to
            format: Format to use ('pickle', 'npz')
        """
        filepath = self.base_dir / filename
        
        if format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(embeddings, f)
        elif format == 'npz':
            np.savez_compressed(filepath, **embeddings)
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self.logger.info(f"Saved embeddings to {filepath}")
        
    def load_embeddings(self, 
                       filename: str,
                       format: str = 'pickle') -> Dict[str, np.ndarray]:
        """Load embeddings from disk.
        
        Args:
            filename: Filename to load from
            format: Format to use ('pickle', 'npz')
            
        Returns:
            Dictionary mapping language names to embedding matrices
        """
        filepath = self.base_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        if format == 'pickle':
            with open(filepath, 'rb') as f:
                embeddings = pickle.load(f)
        elif format == 'npz':
            data = np.load(filepath)
            embeddings = {key: data[key] for key in data.files}
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self.logger.info(f"Loaded embeddings from {filepath}")
        return embeddings
        
    def save_distance_matrix(self, 
                            distance_matrix: Dict[str, Dict[str, float]],
                            filename: str,
                            format: str = 'json') -> None:
        """Save distance matrix to disk.
        
        Args:
            distance_matrix: Distance matrix as nested dictionary
            filename: Filename to save to
            format: Format to use ('json', 'csv', 'pickle')
        """
        filepath = self.base_dir / filename
        
        if format == 'json':
            # Handle non-serializable values
            cleaned_matrix = {}
            for lang1, distances in distance_matrix.items():
                cleaned_matrix[lang1] = {}
                for lang2, dist in distances.items():
                    if np.isfinite(dist):
                        cleaned_matrix[lang1][lang2] = float(dist)
                    elif np.isinf(dist):
                        cleaned_matrix[lang1][lang2] = "inf"
                    else:
                        cleaned_matrix[lang1][lang2] = "nan"
                        
            with open(filepath, 'w') as f:
                json.dump(cleaned_matrix, f, indent=2)
                
        elif format == 'csv':
            # Convert to DataFrame and save as CSV
            languages = list(distance_matrix.keys())
            matrix_data = []
            
            for lang1 in languages:
                row = []
                for lang2 in languages:
                    value = distance_matrix[lang1].get(lang2, np.nan)
                    row.append(value)
                matrix_data.append(row)
                
            df = pd.DataFrame(matrix_data, index=languages, columns=languages)
            df.to_csv(filepath)
            
        elif format == 'pickle':
            with open(filepath, 'wb') as f:
                pickle.dump(distance_matrix, f)
                
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self.logger.info(f"Saved distance matrix to {filepath}")
        
    def load_distance_matrix(self, 
                            filename: str,
                            format: str = 'json') -> Dict[str, Dict[str, float]]:
        """Load distance matrix from disk.
        
        Args:
            filename: Filename to load from
            format: Format to use ('json', 'csv', 'pickle')
            
        Returns:
            Distance matrix as nested dictionary
        """
        filepath = self.base_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        if format == 'json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                
            # Convert string representations back to floats
            distance_matrix = {}
            for lang1, distances in data.items():
                distance_matrix[lang1] = {}
                for lang2, dist in distances.items():
                    if dist == "inf":
                        distance_matrix[lang1][lang2] = np.inf
                    elif dist == "nan":
                        distance_matrix[lang1][lang2] = np.nan
                    else:
                        distance_matrix[lang1][lang2] = float(dist)
                        
        elif format == 'csv':
            df = pd.read_csv(filepath, index_col=0)
            distance_matrix = {}
            for lang1 in df.index:
                distance_matrix[lang1] = {}
                for lang2 in df.columns:
                    distance_matrix[lang1][lang2] = df.loc[lang1, lang2]
                    
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                distance_matrix = pickle.load(f)
                
        else:
            raise ValueError(f"Unknown format: {format}")
            
        self.logger.info(f"Loaded distance matrix from {filepath}")
        return distance_matrix
        
    def save_results(self, 
                    results: Dict[str, Any],
                    filename: str,
                    include_metadata: bool = True) -> None:
        """Save analysis results to disk.
        
        Args:
            results: Dictionary of results
            filename: Filename to save to
            include_metadata: Whether to include metadata
        """
        filepath = self.base_dir / filename
        
        if include_metadata:
            results_with_metadata = {
                'metadata': {
                    'timestamp': datetime.now().isoformat(),
                    'version': '0.1.0'
                },
                'results': results
            }
        else:
            results_with_metadata = results
            
        # Recursively clean non-serializable values
        cleaned_results = self._clean_for_json(results_with_metadata)
        
        with open(filepath, 'w') as f:
            json.dump(cleaned_results, f, indent=2)
            
        self.logger.info(f"Saved results to {filepath}")
        
    def _clean_for_json(self, obj: Any) -> Any:
        """Recursively clean object for JSON serialization."""
        if isinstance(obj, dict):
            return {key: self._clean_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._clean_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            if np.isfinite(obj):
                return float(obj)
            elif np.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            else:
                return "nan"
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        else:
            return obj
            
    def load_results(self, filename: str) -> Dict[str, Any]:
        """Load analysis results from disk.
        
        Args:
            filename: Filename to load from
            
        Returns:
            Dictionary of results
        """
        filepath = self.base_dir / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"File not found: {filepath}")
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        self.logger.info(f"Loaded results from {filepath}")
        return data
        
    def export_to_csv(self, 
                     data: Union[Dict[str, Dict[str, float]], pd.DataFrame],
                     filename: str,
                     index_name: str = 'Language') -> None:
        """Export data to CSV format.
        
        Args:
            data: Data to export (nested dict or DataFrame)
            filename: Filename to save to
            index_name: Name for the index column
        """
        filepath = self.base_dir / filename
        
        if isinstance(data, dict):
            # Convert nested dict to DataFrame
            df = pd.DataFrame(data).T
        else:
            df = data.copy()
            
        df.index.name = index_name
        df.to_csv(filepath)
        
        self.logger.info(f"Exported data to CSV: {filepath}")
        
    def create_summary_report(self, 
                            results: Dict[str, Any],
                            filename: str = "summary_report.txt") -> None:
        """Create a human-readable summary report.
        
        Args:
            results: Analysis results
            filename: Filename for the report
        """
        filepath = self.base_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("LINGUISTIC DISTANCE ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Write summary statistics
            if 'distance_matrices' in results:
                f.write("DISTANCE MATRICES:\n")
                f.write("-" * 20 + "\n")
                
                for method_name, matrix in results['distance_matrices'].items():
                    f.write(f"\n{method_name.upper()}:\n")
                    
                    languages = list(matrix.keys())
                    
                    # Find min/max/avg distances
                    all_distances = []
                    for lang1 in languages:
                        for lang2 in languages:
                            if lang1 != lang2:
                                dist = matrix[lang1].get(lang2, np.nan)
                                if np.isfinite(dist):
                                    all_distances.append(dist)
                                    
                    if all_distances:
                        f.write(f"  Min distance: {min(all_distances):.4f}\n")
                        f.write(f"  Max distance: {max(all_distances):.4f}\n")
                        f.write(f"  Avg distance: {np.mean(all_distances):.4f}\n")
                        f.write(f"  Std distance: {np.std(all_distances):.4f}\n")
                        
                    # Most similar and dissimilar pairs
                    if len(all_distances) > 0:
                        min_dist = min(all_distances)
                        max_dist = max(all_distances)
                        
                        min_pairs = []
                        max_pairs = []
                        
                        for lang1 in languages:
                            for lang2 in languages:
                                if lang1 != lang2:
                                    dist = matrix[lang1].get(lang2, np.nan)
                                    if np.isfinite(dist):
                                        if abs(dist - min_dist) < 1e-6:
                                            min_pairs.append((lang1, lang2))
                                        if abs(dist - max_dist) < 1e-6:
                                            max_pairs.append((lang1, lang2))
                                            
                        if min_pairs:
                            f.write(f"  Most similar: {min_pairs[0]} ({min_dist:.4f})\n")
                        if max_pairs:
                            f.write(f"  Most dissimilar: {max_pairs[0]} ({max_dist:.4f})\n")
                            
            # Write embedding statistics if available
            if 'embedding_stats' in results:
                f.write("\n\nEMBEDDING STATISTICS:\n")
                f.write("-" * 20 + "\n")
                
                for lang, stats in results['embedding_stats'].items():
                    f.write(f"\n{lang.upper()}:\n")
                    for stat_name, value in stats.items():
                        if isinstance(value, (int, float)):
                            f.write(f"  {stat_name}: {value:.4f}\n")
                        else:
                            f.write(f"  {stat_name}: {value}\n")
                            
        self.logger.info(f"Created summary report: {filepath}")
        
    def get_file_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about saved files.
        
        Returns:
            Dictionary with file information
        """
        file_info = {}
        
        for filepath in self.base_dir.iterdir():
            if filepath.is_file():
                stats = filepath.stat()
                file_info[filepath.name] = {
                    'size_bytes': stats.st_size,
                    'size_mb': stats.st_size / (1024 * 1024),
                    'modified': datetime.fromtimestamp(stats.st_mtime).isoformat(),
                    'extension': filepath.suffix
                }
                
        return file_info


# Convenience functions
def save_embeddings(embeddings: Dict[str, np.ndarray],
                   filename: str,
                   format: str = 'pickle',
                   base_dir: str = "results") -> None:
    """Convenience function to save embeddings."""
    io_utils = IOUtils(base_dir)
    io_utils.save_embeddings(embeddings, filename, format)


def load_embeddings(filename: str,
                   format: str = 'pickle',
                   base_dir: str = "results") -> Dict[str, np.ndarray]:
    """Convenience function to load embeddings."""
    io_utils = IOUtils(base_dir)
    return io_utils.load_embeddings(filename, format)


def save_results(results: Dict[str, Any],
                filename: str,
                base_dir: str = "results",
                include_metadata: bool = True) -> None:
    """Convenience function to save results."""
    io_utils = IOUtils(base_dir)
    io_utils.save_results(results, filename, include_metadata)