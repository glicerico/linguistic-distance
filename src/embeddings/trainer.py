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
    """Callback to track training progress with early stopping and best model saving."""
    
    def __init__(self, patience=5, min_delta=0.001, save_path=None, language=None):
        self.epoch = 0
        self.losses = []
        self.previous_loss = 0
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        self.should_stop = False
        self.save_path = save_path
        self.language = language
        self.best_model = None
        
    def on_epoch_end(self, model):
        # Get cumulative loss and calculate per-epoch loss
        cumulative_loss = model.get_latest_training_loss()
        epoch_loss = cumulative_loss - self.previous_loss
        
        self.losses.append(epoch_loss)
        
        # Check if this is the best model so far
        if epoch_loss < self.best_loss - self.min_delta:
            self.best_loss = epoch_loss
            self.wait = 0
            
            # Save the best model if save_path is provided
            if self.save_path and self.language:
                self._save_best_model(model)
                print(f'💾 New best model saved at epoch {self.epoch} (loss: {epoch_loss:.1f})')
        else:
            self.wait += 1
            
        # Print progress
        if self.epoch % 5 == 0 or self.epoch < 5:  # Print every 5 epochs after first 5
            print(f'Epoch {self.epoch}: loss = {epoch_loss:.1f} (best: {self.best_loss:.1f}, patience: {self.wait}/{self.patience})')
        
        # Check if we should stop early
        if self.wait >= self.patience and self.epoch >= 10:  # Don't stop too early
            print(f'Early stopping at epoch {self.epoch} (no improvement for {self.patience} epochs)')
            self.should_stop = True
        
        self.previous_loss = cumulative_loss
        self.epoch += 1
    
    def _save_best_model(self, model):
        """Save the current best model."""
        if not self.save_path or not self.language:
            return
        
        try:
            # Determine model type from model class
            model_type = "fasttext" if hasattr(model, 'min_n') else "word2vec"
            
            # Save the full model
            model_path = self.save_path / f"{self.language}_{model_type}_best.model"
            model.save(str(model_path))
            
            # Save just the embeddings
            embeddings_path = self.save_path / f"{self.language}_{model_type}_best_embeddings.pkl"
            embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
        except Exception as e:
            print(f'⚠️  Warning: Could not save best model: {e}')


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
                       vector_size: int = 150,
                       window: int = 5,
                       min_count: int = 4,
                       workers: int = 4,
                       epochs: int = 40,
                       sg: int = 0,
                       sample: float = 1e-3) -> Word2Vec:
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
        
        # Initialize callback with best model saving
        callback = TrainingCallback(save_path=self.output_dir, language=language)
        
        # Train model with improved parameters
        model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            epochs=epochs,
            sg=sg,
            callbacks=[callback],
            compute_loss=True,
            alpha=0.025,        # Initial learning rate
            min_alpha=0.0001,   # Minimum learning rate
            negative=5,         # Negative sampling
            sample=sample       # Subsampling threshold
        )
        
        # Use best model as the final model if available
        best_model_path = self.output_dir / f"{language}_word2vec_best.model"
        best_embeddings_path = self.output_dir / f"{language}_word2vec_best_embeddings.pkl"
        
        if best_model_path.exists() and best_embeddings_path.exists():
            # Copy best model to standard location
            final_model_path = self.output_dir / f"{language}_word2vec.model"
            final_embeddings_path = self.output_dir / f"{language}_word2vec_embeddings.pkl"
            
            import shutil
            shutil.copy(str(best_model_path), str(final_model_path))
            shutil.copy(str(best_embeddings_path), str(final_embeddings_path))
            
            self.logger.info(f"✅ Using best model (loss: {callback.best_loss:.1f}) as final model")
            self.logger.info(f"Saved to {final_model_path}")
        else:
            # Fallback: save current model
            model_path = self.output_dir / f"{language}_word2vec.model"
            model.save(str(model_path))
            
            embeddings_path = self.output_dir / f"{language}_word2vec_embeddings.pkl"
            embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
            self.logger.info(f"Saved final model to {model_path}")
            
        self.logger.info(f"Vocabulary size: {len(model.wv)}")
        
        return model
        
    def train_fasttext(self, 
                       corpus_file: str,
                       language: str,
                       vector_size: int = 150,
                       window: int = 5,
                       min_count: int = 4,
                       workers: int = 4,
                       epochs: int = 40,
                       sg: int = 0,
                       min_n: int = 3,
                       max_n: int = 6,
                       sample: float = 1e-3) -> FastText:
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
        
        # Initialize callback with best model saving
        callback = TrainingCallback(save_path=self.output_dir, language=language)
        
        # Train model with improved parameters
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
            compute_loss=True,
            alpha=0.025,        # Initial learning rate
            min_alpha=0.0001,   # Minimum learning rate
            negative=5,         # Negative sampling
            sample=sample       # Subsampling threshold
        )
        
        # Use best model as the final model if available
        best_model_path = self.output_dir / f"{language}_fasttext_best.model"
        best_embeddings_path = self.output_dir / f"{language}_fasttext_best_embeddings.pkl"
        
        if best_model_path.exists() and best_embeddings_path.exists():
            # Copy best model to standard location
            final_model_path = self.output_dir / f"{language}_fasttext.model"
            final_embeddings_path = self.output_dir / f"{language}_fasttext_embeddings.pkl"
            
            import shutil
            shutil.copy(str(best_model_path), str(final_model_path))
            shutil.copy(str(best_embeddings_path), str(final_embeddings_path))
            
            self.logger.info(f"✅ Using best model (loss: {callback.best_loss:.1f}) as final model")
            self.logger.info(f"Saved to {final_model_path}")
        else:
            # Fallback: save current model
            model_path = self.output_dir / f"{language}_fasttext.model"
            model.save(str(model_path))
            
            embeddings_path = self.output_dir / f"{language}_fasttext_embeddings.pkl"
            embeddings = {word: model.wv[word] for word in model.wv.index_to_key}
            with open(embeddings_path, 'wb') as f:
                pickle.dump(embeddings, f)
                
            self.logger.info(f"Saved final model to {model_path}")
            
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