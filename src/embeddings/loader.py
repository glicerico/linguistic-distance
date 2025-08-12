"""Embedding loading and management utilities."""

import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set
from gensim.models import Word2Vec, FastText
import logging


class EmbeddingLoader:
    """Load and manage trained embeddings."""
    
    def __init__(self, embeddings_dir: str = "data/embeddings"):
        """Initialize the loader.
        
        Args:
            embeddings_dir: Directory containing trained embeddings
        """
        self.embeddings_dir = Path(embeddings_dir)
        self.loaded_embeddings = {}
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_embeddings(self, language: str, model_type: str = "word2vec") -> Dict[str, np.ndarray]:
        """Load embeddings for a specific language.
        
        Args:
            language: Language identifier
            model_type: Type of embeddings ('word2vec' or 'fasttext')
            
        Returns:
            Dictionary mapping words to vectors
        """
        cache_key = f"{language}_{model_type}"
        
        if cache_key in self.loaded_embeddings:
            return self.loaded_embeddings[cache_key]
            
        embeddings_file = self.embeddings_dir / f"{language}_{model_type}_embeddings.pkl"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            
        self.loaded_embeddings[cache_key] = embeddings
        self.logger.info(f"Loaded {len(embeddings)} embeddings for {language}")
        
        return embeddings
        
    def load_model(self, language: str, model_type: str = "word2vec"):
        """Load a trained model.
        
        Args:
            language: Language identifier
            model_type: Type of model
            
        Returns:
            Loaded model (Word2Vec or FastText)
        """
        model_file = self.embeddings_dir / f"{language}_{model_type}.model"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
            
        if model_type == "word2vec":
            return Word2Vec.load(str(model_file))
        elif model_type == "fasttext":
            return FastText.load(str(model_file))
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
    def get_common_vocabulary(self, 
                             language1: str, 
                             language2: str, 
                             model_type: str = "word2vec",
                             min_frequency: int = 1) -> Set[str]:
        """Get common vocabulary between two languages.
        
        Args:
            language1: First language
            language2: Second language
            model_type: Type of embeddings
            min_frequency: Minimum frequency threshold
            
        Returns:
            Set of common words
        """
        emb1 = self.load_embeddings(language1, model_type)
        emb2 = self.load_embeddings(language2, model_type)
        
        vocab1 = set(emb1.keys())
        vocab2 = set(emb2.keys())
        
        common_vocab = vocab1.intersection(vocab2)
        
        self.logger.info(f"Found {len(common_vocab)} common words between {language1} and {language2}")
        
        return common_vocab
        
    def align_embeddings(self, 
                        language1: str, 
                        language2: str, 
                        model_type: str = "word2vec",
                        common_vocab: Optional[Set[str]] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Align embeddings from two languages using common vocabulary.
        
        Args:
            language1: First language
            language2: Second language
            model_type: Type of embeddings
            common_vocab: Common vocabulary (if None, will compute)
            
        Returns:
            Tuple of (aligned_matrix1, aligned_matrix2, word_list)
        """
        emb1 = self.load_embeddings(language1, model_type)
        emb2 = self.load_embeddings(language2, model_type)
        
        if common_vocab is None:
            common_vocab = self.get_common_vocabulary(language1, language2, model_type)
            
        # Sort words for consistent ordering
        common_words = sorted(list(common_vocab))
        
        if len(common_words) == 0:
            raise ValueError(f"No common vocabulary found between {language1} and {language2}")
            
        # Create aligned matrices
        matrix1 = np.array([emb1[word] for word in common_words])
        matrix2 = np.array([emb2[word] for word in common_words])
        
        self.logger.info(f"Aligned embeddings: {matrix1.shape[0]} words, {matrix1.shape[1]} dimensions")
        
        return matrix1, matrix2, common_words
        
    def get_embedding_statistics(self, language: str, model_type: str = "word2vec") -> Dict[str, float]:
        """Get statistics about embeddings.
        
        Args:
            language: Language identifier
            model_type: Type of embeddings
            
        Returns:
            Dictionary of statistics
        """
        embeddings = self.load_embeddings(language, model_type)
        
        # Convert to matrix
        words = list(embeddings.keys())
        matrix = np.array([embeddings[word] for word in words])
        
        # Calculate statistics
        stats = {
            'vocabulary_size': len(words),
            'vector_dimension': matrix.shape[1],
            'mean_vector_norm': float(np.mean(np.linalg.norm(matrix, axis=1))),
            'std_vector_norm': float(np.std(np.linalg.norm(matrix, axis=1))),
            'mean_component': float(np.mean(matrix)),
            'std_component': float(np.std(matrix)),
            'min_component': float(np.min(matrix)),
            'max_component': float(np.max(matrix))
        }
        
        return stats
        
    def find_similar_words(self, 
                          word: str, 
                          language: str, 
                          model_type: str = "word2vec",
                          top_k: int = 10) -> List[Tuple[str, float]]:
        """Find most similar words to a given word.
        
        Args:
            word: Target word
            language: Language to search in
            model_type: Type of embeddings
            top_k: Number of similar words to return
            
        Returns:
            List of (word, similarity) tuples
        """
        embeddings = self.load_embeddings(language, model_type)
        
        if word not in embeddings:
            return []
            
        target_vector = embeddings[word]
        similarities = []
        
        for other_word, other_vector in embeddings.items():
            if other_word != word:
                # Calculate cosine similarity
                similarity = np.dot(target_vector, other_vector) / (
                    np.linalg.norm(target_vector) * np.linalg.norm(other_vector)
                )
                similarities.append((other_word, float(similarity)))
                
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
        
    def save_aligned_embeddings(self, 
                               language1: str, 
                               language2: str, 
                               matrix1: np.ndarray, 
                               matrix2: np.ndarray, 
                               words: List[str],
                               model_type: str = "word2vec") -> None:
        """Save aligned embeddings to disk.
        
        Args:
            language1: First language
            language2: Second language
            matrix1: Aligned embeddings for language1
            matrix2: Aligned embeddings for language2
            words: List of aligned words
            model_type: Type of embeddings
        """
        # Create aligned embeddings data
        aligned_data = {
            'language1': language1,
            'language2': language2,
            'matrix1': matrix1,
            'matrix2': matrix2,
            'words': words,
            'model_type': model_type
        }
        
        # Save to file
        output_file = self.embeddings_dir / f"{language1}_{language2}_{model_type}_aligned.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(aligned_data, f)
            
        self.logger.info(f"Saved aligned embeddings to {output_file}")
        
    def load_aligned_embeddings(self, 
                               language1: str, 
                               language2: str, 
                               model_type: str = "word2vec") -> Dict[str, any]:
        """Load previously saved aligned embeddings.
        
        Args:
            language1: First language
            language2: Second language
            model_type: Type of embeddings
            
        Returns:
            Dictionary with aligned data
        """
        input_file = self.embeddings_dir / f"{language1}_{language2}_{model_type}_aligned.pkl"
        
        if not input_file.exists():
            # Try reverse order
            input_file = self.embeddings_dir / f"{language2}_{language1}_{model_type}_aligned.pkl"
            
        if not input_file.exists():
            raise FileNotFoundError(f"Aligned embeddings not found for {language1}-{language2}")
            
        with open(input_file, 'rb') as f:
            aligned_data = pickle.load(f)
            
        return aligned_data
        
    def get_available_embeddings(self) -> Dict[str, List[str]]:
        """Get information about available embeddings.
        
        Returns:
            Dictionary mapping languages to available model types
        """
        available = {}
        
        for embeddings_file in self.embeddings_dir.glob("*_embeddings.pkl"):
            parts = embeddings_file.stem.split('_')
            if len(parts) >= 3:
                language = parts[0]
                model_type = parts[1]
                
                if language not in available:
                    available[language] = []
                    
                if model_type not in available[language]:
                    available[language].append(model_type)
                    
        return available