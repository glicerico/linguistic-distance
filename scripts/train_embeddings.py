#!/usr/bin/env python3
"""Train word embeddings on preprocessed multilingual data."""

import sys
import argparse
from pathlib import Path
import logging
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from embeddings.trainer import EmbeddingTrainer


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function for training embeddings."""
    parser = argparse.ArgumentParser(description="Train word embeddings on multilingual data")
    
    parser.add_argument(
        "--languages", 
        nargs="+", 
        default=["english", "spanish", "german", "italian", "dutch"],
        help="Languages to train embeddings for"
    )
    parser.add_argument(
        "--input-dir", 
        default="data/processed",
        help="Directory containing processed text files"
    )
    parser.add_argument(
        "--output-dir", 
        default="data/embeddings",
        help="Directory to save trained embeddings"
    )
    parser.add_argument(
        "--model-type", 
        choices=["word2vec", "fasttext"],
        default="word2vec",
        help="Type of embeddings to train"
    )
    parser.add_argument(
        "--vector-size", 
        type=int, 
        default=150,
        help="Dimensionality of embeddings"
    )
    parser.add_argument(
        "--window", 
        type=int, 
        default=5,
        help="Context window size"
    )
    parser.add_argument(
        "--min-count", 
        type=int, 
        default=4,
        help="Minimum word frequency"
    )
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=40,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--workers", 
        type=int, 
        default=4,
        help="Number of worker threads"
    )
    parser.add_argument(
        "--sg", 
        type=int, 
        choices=[0, 1],
        default=0,
        help="Training algorithm: 0=CBOW, 1=Skip-gram"
    )
    parser.add_argument(
        "--save-evaluation", 
        action="store_true",
        help="Save evaluation results to JSON"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    # FastText-specific arguments
    parser.add_argument(
        "--min-n", 
        type=int, 
        default=3,
        help="Minimum character n-gram length (FastText only)"
    )
    parser.add_argument(
        "--max-n", 
        type=int, 
        default=6,
        help="Maximum character n-gram length (FastText only)"
    )
    parser.add_argument(
        "--sample", 
        type=float, 
        default=1e-3,
        help="Subsampling threshold for frequent words"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Prepare training arguments
    training_args = {
        'vector_size': args.vector_size,
        'window': args.window,
        'min_count': args.min_count,
        'workers': args.workers,
        'epochs': args.epochs,
        'sg': args.sg,
        'sample': args.sample
    }
    
    if args.model_type == 'fasttext':
        training_args.update({
            'min_n': args.min_n,
            'max_n': args.max_n
        })
    
    logger.info(f"Training {args.model_type} embeddings with parameters:")
    for key, value in training_args.items():
        logger.info(f"  {key}: {value}")
    
    # Initialize trainer
    trainer = EmbeddingTrainer(args.output_dir)
    
    # Train embeddings for all languages
    logger.info(f"Training embeddings for languages: {args.languages}")
    
    models = trainer.train_all_languages(
        input_dir=args.input_dir,
        languages=args.languages,
        model_type=args.model_type,
        **training_args
    )
    
    # Evaluate trained models
    evaluations = {}
    for language, model in models.items():
        logger.info(f"Evaluating {language} embeddings...")
        evaluation = trainer.evaluate_embeddings(model, language)
        evaluations[language] = evaluation
        
        logger.info(f"{language} evaluation:")
        logger.info(f"  Vocabulary size: {evaluation['vocabulary_size']}")
        logger.info(f"  Vector dimension: {evaluation['vector_dimension']}")
        logger.info(f"  Average vector norm: {evaluation['average_vector_norm']:.4f}")
        
        # Show some sample similarities
        if evaluation['sample_similarities']:
            logger.info("  Sample word similarities:")
            for word_pair, similarity in list(evaluation['sample_similarities'].items())[:3]:
                logger.info(f"    {word_pair}: {similarity:.4f}")
    
    # Save evaluation results
    if args.save_evaluation:
        eval_file = Path(args.output_dir) / f"{args.model_type}_evaluation.json"
        with open(eval_file, 'w') as f:
            json.dump(evaluations, f, indent=2)
        logger.info(f"Saved evaluation results to {eval_file}")
    
    # Show model info
    model_info = trainer.get_model_info()
    logger.info("\\nTrained models:")
    for language, models_info in model_info.items():
        for model_type, info in models_info.items():
            size_mb = info['file_size'] / (1024 * 1024)
            logger.info(f"  {language} {model_type}: {size_mb:.1f} MB")
    
    logger.info("Embedding training complete!")


if __name__ == "__main__":
    main()