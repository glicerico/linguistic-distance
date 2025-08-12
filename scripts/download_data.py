#!/usr/bin/env python3
"""Download multilingual Bible data for training embeddings."""

import sys
import argparse
from pathlib import Path
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.downloader import BibleDownloader
from data.preprocessor import TextPreprocessor


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main function for downloading and preprocessing data."""
    parser = argparse.ArgumentParser(description="Download multilingual Bible data")
    
    parser.add_argument(
        "--languages", 
        nargs="+", 
        default=["english", "spanish", "german", "italian", "dutch"],
        help="Languages to download (default: all supported languages)"
    )
    parser.add_argument(
        "--raw-dir", 
        default="data/raw",
        help="Directory for raw data (default: data/raw)"
    )
    parser.add_argument(
        "--processed-dir", 
        default="data/processed", 
        help="Directory for processed data (default: data/processed)"
    )
    parser.add_argument(
        "--force-download", 
        action="store_true",
        help="Re-download existing files"
    )
    parser.add_argument(
        "--skip-preprocessing", 
        action="store_true",
        help="Skip preprocessing step"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    logger = logging.getLogger(__name__)
    
    # Download data
    logger.info("Starting data download...")
    downloader = BibleDownloader(args.raw_dir)
    
    download_results = downloader.download_all(
        languages=args.languages,
        force_download=args.force_download
    )
    
    # Report download results
    successful_downloads = [lang for lang, success in download_results.items() if success]
    failed_downloads = [lang for lang, success in download_results.items() if not success]
    
    logger.info(f"Successfully downloaded: {successful_downloads}")
    if failed_downloads:
        logger.warning(f"Failed downloads: {failed_downloads}")
        
    # Get file information
    file_info = downloader.get_file_info()
    for lang, info in file_info.items():
        if 'num_words' in info:
            logger.info(f"{lang}: {info['num_words']} words, {info['num_lines']} lines")
            
    # Preprocessing
    if not args.skip_preprocessing:
        logger.info("Starting preprocessing...")
        preprocessor = TextPreprocessor()
        
        preprocessing_results = preprocessor.preprocess_all_languages(
            input_dir=args.raw_dir,
            output_dir=args.processed_dir,
            languages=successful_downloads
        )
        
        # Report preprocessing results
        for lang, stats in preprocessing_results.items():
            logger.info(f"Processed {lang}: {stats['num_sentences']} sentences, "
                       f"{stats['num_words']} words, {stats['num_unique_words']} unique")
    else:
        logger.info("Skipping preprocessing as requested")
        
    logger.info("Data download and preprocessing complete!")


if __name__ == "__main__":
    main()