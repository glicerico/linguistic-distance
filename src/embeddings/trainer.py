"""Embedding training utilities for monolingual corpora."""

import os
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import numpy as np
from gensim.models import Word2Vec, FastText
from gensim.models.callbacks import CallbackAny2Vec
import pickle
from tqdm import tqdm


class TrainingCallback(CallbackAny2Vec):
    """Callback to track training progress."""
    
    def __init__(self):
        self.epoch = 0
        self.losses = []
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        self.losses.append(loss)
        print(f'Loss after epoch {self.epoch}: {loss}')
        self.epoch += 1


class EmbeddingTrainer:
    """Train word embeddings on monolingual corpora."""
    
    def __init__(self, output_dir: str = "data/embeddings"):
        """Initialize the trainer.
        
        Args:
            output_dir: Directory to save trained embeddings
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_corpus(self, file_path: str) -> List[List[str]]:
        """Load a text corpus for training.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            List of tokenized sentences
        """
        sentences = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    words = line.split()
                    if len(words) >= 2:  # Keep sentences with at least 2 words
                        sentences.append(words)
                        
        self.logger.info(f"Loaded {len(sentences)} sentences from {file_path}")
        return sentences
        
    def train_word2vec(self, 
                       corpus_file: str,
                       language: str,
                       vector_size: int = 100,
                       window: int = 5,
                       min_count: int = 5,
                       workers: int = 4,
                       epochs: int = 100,
                       sg: int = 0) -> Word2Vec:
        """Train Word2Vec embeddings.
        
        Args:
            corpus_file: Path to preprocessed corpus file
            language: Language identifier
            vector_size: Dimensionality of embeddings
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Training epochs
            sg: Skip-gram (1) or CBOW (0)
            
        Returns:
            Trained Word2Vec model
        """
        self.logger.info(f"Training Word2Vec for {language}")
        
        # Load corpus
        sentences = self.load_corpus(corpus_file)
        
        # Initialize callback
        callback = TrainingCallback()
        
        # Train model
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=sg,
            callbacks=[callback],
            compute_loss=True
        )
        
        # Save model
        model_path = self.output_dir / f"{language}_word2vec.model"
        model.save(str(model_path))
        
        # Save just the embeddings
        embeddings_path = self.output_dir / f"{language}_word2vec_embeddings.pkl"
        embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        self.logger.info(f"Saved Word2Vec model to {model_path}")
        self.logger.info(f"Vocabulary size: {len(model.wv)}")
        
        return model
        
    def train_fasttext(self, 
                       corpus_file: str,
                       language: str,
                       vector_size: int = 100,
                       window: int = 5,
                       min_count: int = 5,
                       workers: int = 4,
                       epochs: int = 100,
                       sg: int = 0,
                       min_n: int = 3,
                       max_n: int = 6) -> FastText:
        """Train FastText embeddings.
        
        Args:
            corpus_file: Path to preprocessed corpus file
            language: Language identifier
            vector_size: Dimensionality of embeddings
            window: Context window size
            min_count: Minimum word frequency
            workers: Number of worker threads
            epochs: Training epochs
            sg: Skip-gram (1) or CBOW (0)
            min_n: Minimum character n-gram length
            max_n: Maximum character n-gram length
            
        Returns:
            Trained FastText model
        """
        self.logger.info(f"Training FastText for {language}")
        
        # Load corpus
        sentences = self.load_corpus(corpus_file)
        
        # Initialize callback
        callback = TrainingCallback()
        
        # Train model
        model = FastText(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=sg,
            min_n=min_n,
            max_n=max_n,
            callbacks=[callback],
            compute_loss=True
        )
        
        # Save model
        model_path = self.output_dir / f"{language}_fasttext.model"
        model.save(str(model_path))
        
        # Save embeddings
        embeddings_path = self.output_dir / f"{language}_fasttext_embeddings.pkl"
        embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
            
        self.logger.info(f"Saved FastText model to {model_path}")
        self.logger.info(f"Vocabulary size: {len(model.wv)}")
        
        return model
        
    def train_all_languages(self,
                           input_dir: str = "data/processed",
                           languages: Optional[List[str]] = None,
                           model_type: str = "word2vec",
                           **kwargs) -> Dict[str, Any]:
        """Train embeddings for multiple languages.
        
        Args:
            input_dir: Directory containing preprocessed files
            languages: List of languages to train
            model_type: Type of embeddings ('word2vec' or 'fasttext')
            **kwargs: Additional arguments for training
            
        Returns:
            Dictionary of trained models
        """
        if languages is None:
            languages = ['english', 'spanish', 'german', 'italian', 'dutch']
            
        models = {}
        
        for language in tqdm(languages, desc=f"Training {model_type} embeddings"):
            input_file = Path(input_dir) / f"{language}_processed.txt"
            
            if not input_file.exists():
                self.logger.warning(f"Input file not found for {language}: {input_file}")
                continue
                
            try:
                if model_type == "word2vec":
                    model = self.train_word2vec(str(input_file), language, **kwargs)
                elif model_type == "fasttext":
                    model = self.train_fasttext(str(input_file), language, **kwargs)
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                    
                models[language] = model
                
            except Exception as e:
                self.logger.error(f"Failed to train {model_type} for {language}: {e}")
                
        return models
        
    def evaluate_embeddings(self, model: Union[Word2Vec, FastText], language: str) -> Dict[str, Any]:
        """Evaluate trained embeddings.
        
        Args:
            model: Trained embedding model
            language: Language identifier
            
        Returns:
            Dictionary of evaluation metrics
        """
        vocab_size = len(model.wv)
        
        # Get some sample similarities
        sample_words = list(model.wv.index_to_key[:10])
        similarities = {}
        
        if len(sample_words) >= 2:
            for i, word1 in enumerate(sample_words[:5]):
                for word2 in sample_words[i+1:i+3]:
                    try:
                        sim = model.wv.similarity(word1, word2)
                        similarities[f"{word1}-{word2}"] = float(sim)
                    except KeyError:
                        pass
                        
        # Calculate average vector norm
        vectors = [model.wv[word] for word in sample_words]
        avg_norm = float(np.mean([np.linalg.norm(vec) for vec in vectors]))
        
        return {
            'vocabulary_size': vocab_size,
            'vector_dimension': model.wv.vector_size,
            'sample_similarities': similarities,
            'average_vector_norm': avg_norm,
            'most_frequent_words': sample_words
        }
        
    def get_model_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about saved models.
        
        Returns:
            Dictionary with model information
        """
        info = {}
        
        for model_file in self.output_dir.glob("*.model"):
            language = model_file.stem.split('_')[0]
            model_type = model_file.stem.split('_')[1]
            
            if language not in info:
                info[language] = {}
                
            info[language][model_type] = {
                'file_path': str(model_file),
                'file_size': model_file.stat().st_size,
                'exists': True
            }
            
        return info
        
    def load_embeddings_dict(self, language: str, model_type: str = "word2vec") -> Dict[str, np.ndarray]:
        """Load embeddings as a dictionary.
        
        Args:
            language: Language identifier
            model_type: Type of embeddings
            
        Returns:
            Dictionary mapping words to vectors
        """
        embeddings_file = self.output_dir / f"{language}_{model_type}_embeddings.pkl"
        
        if not embeddings_file.exists():
            raise FileNotFoundError(f"Embeddings file not found: {embeddings_file}")
            
        with open(embeddings_file, 'rb') as f:
            embeddings = pickle.load(f)
            
        return embeddings