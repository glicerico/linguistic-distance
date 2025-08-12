"""Text preprocessing utilities for multilingual corpora."""

import re
import string
from pathlib import Path
from typing import List, Dict, Optional, Callable
import unicodedata
from collections import Counter


class TextPreprocessor:
    """Preprocess text data for embedding training."""
    
    def __init__(self):
        """Initialize the preprocessor."""
        # Common preprocessing patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b')
        self.verse_pattern = re.compile(r'^\\d+:\\d+\\s*')  # Bible verse numbers like "1:1 "
        self.chapter_pattern = re.compile(r'^Chapter \\d+|^\\d+ ')  # Chapter headers
        self.book_pattern = re.compile(r'^[A-Z][a-z]+ \\d+:\\d+')  # Book references like "Genesis 1:1"
        
    def clean_text(self, text: str, language: str = 'english') -> str:
        """Clean raw text data.
        
        Args:
            text: Raw text to clean
            language: Language of the text
            
        Returns:
            Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs and emails
        text = self.url_pattern.sub(' ', text)
        text = self.email_pattern.sub(' ', text)
        
        # Remove Bible-specific formatting
        text = self.verse_pattern.sub('', text)
        text = self.chapter_pattern.sub('', text)
        text = self.book_pattern.sub('', text)
        
        # Remove verse numbers in various formats
        text = re.sub(r'^\\d+\\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\\b\\d+:\\d+\\b', '', text)
        text = re.sub(r'\\b\\d+\\.\\d+\\b', '', text)
        
        # Normalize Unicode characters
        text = unicodedata.normalize('NFKD', text)
        
        # Language-specific cleaning
        if language == 'german':
            # Handle German umlauts and ß
            text = text.replace('ä', 'ae').replace('ö', 'oe').replace('ü', 'ue')
            text = text.replace('ß', 'ss')
        elif language == 'spanish':
            # Keep Spanish accents for now, but normalize
            pass
        elif language == 'italian':
            # Keep Italian accents
            pass
        elif language == 'dutch':
            # Handle Dutch special characters
            pass
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text
        
    def tokenize(self, text: str, min_word_length: int = 2, max_word_length: int = 50) -> List[str]:
        """Tokenize text into words.
        
        Args:
            text: Text to tokenize
            min_word_length: Minimum word length to keep
            max_word_length: Maximum word length to keep
            
        Returns:
            List of tokens
        """
        # Split on whitespace and punctuation
        words = re.findall(r'\b[a-zA-ZÀ-ÿ]+\b', text)
        
        # Filter by length
        words = [w for w in words if min_word_length <= len(w) <= max_word_length]
        
        return words
        
    def remove_rare_words(self, words: List[str], min_frequency: int = 5) -> List[str]:
        """Remove rare words from the corpus.
        
        Args:
            words: List of words
            min_frequency: Minimum frequency to keep a word
            
        Returns:
            Filtered list of words
        """
        word_counts = Counter(words)
        return [w for w in words if word_counts[w] >= min_frequency]
        
    def preprocess_file(self, 
                       input_file: str, 
                       output_file: str,
                       language: str = 'english',
                       min_word_length: int = 2,
                       max_word_length: int = 50,
                       min_frequency: int = 5,
                       max_sentences: Optional[int] = None) -> Dict[str, int]:
        """Preprocess a text file.
        
        Args:
            input_file: Path to input file
            output_file: Path to output file
            language: Language of the text
            min_word_length: Minimum word length
            max_word_length: Maximum word length
            min_frequency: Minimum word frequency
            max_sentences: Maximum number of sentences to process
            
        Returns:
            Statistics about the preprocessing
        """
        input_path = Path(input_file)
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Read input file
        with open(input_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        # Clean text
        cleaned_text = self.clean_text(text, language)
        
        # Split into sentences
        sentences = self._split_sentences(cleaned_text)
        
        if max_sentences:
            sentences = sentences[:max_sentences]
            
        # Tokenize each sentence
        all_words = []
        processed_sentences = []
        
        for sentence in sentences:
            words = self.tokenize(sentence, min_word_length, max_word_length)
            if len(words) >= 3:  # Keep sentences with at least 3 words
                all_words.extend(words)
                processed_sentences.append(' '.join(words))
                
        # Remove rare words (but don't filter sentences for small datasets)
        if min_frequency > 1:
            word_counts = Counter(all_words)
            valid_words = set(w for w in word_counts if word_counts[w] >= min_frequency)
            
            # Only filter sentences if we have a substantial vocabulary
            if len(valid_words) >= len(all_words) * 0.5:  # Keep if we retain 50%+ of words
                filtered_sentences = []
                for sentence in processed_sentences:
                    words = sentence.split()
                    filtered_words_in_sent = [w for w in words if w in valid_words]
                    if len(filtered_words_in_sent) >= 3:
                        filtered_sentences.append(' '.join(filtered_words_in_sent))
                processed_sentences = filtered_sentences
            
        # Write output file
        with open(output_path, 'w', encoding='utf-8') as f:
            for sentence in processed_sentences:
                f.write(sentence + '\\n')
                
        # Calculate statistics
        word_counts = Counter(all_words)
        stats = {
            'original_chars': len(text),
            'cleaned_chars': len(cleaned_text),
            'num_sentences': len(processed_sentences),
            'num_words': len(all_words),
            'num_unique_words': len(word_counts),
            'avg_sentence_length': len(all_words) / len(processed_sentences) if processed_sentences else 0
        }
        
        return stats
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences.
        
        Args:
            text: Text to split
            
        Returns:
            List of sentences
        """
        # First try splitting on punctuation
        sentences = re.split(r'[.!?]+', text)
        
        # If no punctuation found, split on newlines (for simple test data)
        if len(sentences) == 1 and '\n' in text:
            sentences = text.split('\n')
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 5:  # Keep sentences with reasonable length (lowered from 10)
                cleaned_sentences.append(sent)
                
        return cleaned_sentences
        
    def preprocess_all_languages(self,
                                input_dir: str = "data/raw",
                                output_dir: str = "data/processed",
                                languages: Optional[List[str]] = None) -> Dict[str, Dict[str, int]]:
        """Preprocess all language files.
        
        Args:
            input_dir: Directory containing raw files
            output_dir: Directory to save processed files
            languages: List of languages to process
            
        Returns:
            Dictionary with preprocessing statistics for each language
        """
        if languages is None:
            languages = ['english', 'spanish', 'german', 'italian', 'dutch']
            
        results = {}
        
        for language in languages:
            input_file = Path(input_dir) / f"{language}_bible.txt"
            output_file = Path(output_dir) / f"{language}_processed.txt"
            
            if input_file.exists():
                print(f"Processing {language}...")
                stats = self.preprocess_file(
                    str(input_file),
                    str(output_file),
                    language=language
                )
                results[language] = stats
                print(f"  {stats['num_sentences']} sentences, {stats['num_words']} words, {stats['num_unique_words']} unique words")
            else:
                print(f"Input file not found for {language}: {input_file}")
                
        return results
        
    def create_vocabulary(self, processed_files: List[str], min_frequency: int = 5) -> Dict[str, int]:
        """Create a vocabulary from processed files.
        
        Args:
            processed_files: List of processed text files
            min_frequency: Minimum frequency for vocabulary inclusion
            
        Returns:
            Dictionary mapping words to frequencies
        """
        word_counts = Counter()
        
        for file_path in processed_files:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                words = text.split()
                word_counts.update(words)
                
        # Filter by minimum frequency
        vocabulary = {word: count for word, count in word_counts.items() 
                     if count >= min_frequency}
        
        return vocabulary