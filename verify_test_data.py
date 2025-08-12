#!/usr/bin/env python3
"""
Simple script to verify test data is properly set up and accessible.
This runs without any external dependencies.
"""

import sys
from pathlib import Path
import logging

def setup_logging():
    """Set up basic logging."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def verify_test_data():
    """Verify test data files exist and are readable."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("VERIFYING TEST DATA")
    logger.info("=" * 50)
    
    test_dir = Path("data/test")
    
    if not test_dir.exists():
        logger.error(f"❌ Test directory not found: {test_dir}")
        logger.info("Run: python setup_test_env.py")
        return False
    
    test_files = list(test_dir.glob("*_test.txt"))
    
    if not test_files:
        logger.error("❌ No test files found")
        logger.info("Run: python setup_test_env.py")
        return False
    
    logger.info(f"✅ Found {len(test_files)} test files:")
    
    total_sentences = 0
    total_words = 0
    
    for test_file in test_files:
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Clean lines and count
            clean_lines = [line.strip() for line in lines if line.strip()]
            words = []
            for line in clean_lines:
                words.extend(line.split())
            
            language = test_file.stem.replace("_test", "")
            logger.info(f"  {language}: {len(clean_lines)} sentences, {len(words)} words")
            
            total_sentences += len(clean_lines)
            total_words += len(words)
            
            # Show first sentence as example
            if clean_lines:
                logger.info(f"    Example: '{clean_lines[0]}'")
                
        except Exception as e:
            logger.error(f"❌ Error reading {test_file}: {e}")
            return False
    
    logger.info(f"\\n✅ Total: {total_sentences} sentences, {total_words} words")
    return True

def test_preprocessing_setup():
    """Test that we can set up directories for preprocessing."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING PREPROCESSING SETUP")
    logger.info("=" * 50)
    
    try:
        import shutil
        
        # Create test directories
        test_dirs = [
            "data/test_raw",
            "data/test_processed", 
            "data/test_embeddings"
        ]
        
        for dir_path in test_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            logger.info(f"✅ Created directory: {dir_path}")
        
        # Copy test files to raw directory
        test_dir = Path("data/test")
        raw_dir = Path("data/test_raw")
        
        copied_files = 0
        for test_file in test_dir.glob("*_test.txt"):
            lang = test_file.stem.replace("_test", "")
            target_file = raw_dir / f"{lang}_bible.txt"
            shutil.copy(test_file, target_file)
            logger.info(f"✅ Copied: {test_file.name} -> {target_file.name}")
            copied_files += 1
        
        logger.info(f"\\n✅ Successfully set up {copied_files} files for testing")
        return True
        
    except Exception as e:
        logger.error(f"❌ Setup failed: {e}")
        return False

def test_basic_file_structure():
    """Test that basic file structure is in place."""
    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("TESTING FILE STRUCTURE")
    logger.info("=" * 50)
    
    required_files = [
        "src/data/__init__.py",
        "src/embeddings/__init__.py",
        "src/distance/__init__.py",
        "scripts/train_embeddings.py",
        "examples/basic_usage.py"
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            logger.info(f"✅ {file_path}")
        else:
            logger.error(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        logger.error(f"Missing {len(missing_files)} files")
        return False
    else:
        logger.info("\\n✅ All required files present")
        return True

def main():
    """Run verification tests."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 LINGUISTIC DISTANCE - TEST DATA VERIFICATION")
    logger.info("=" * 60)
    
    tests = [
        ("File Structure", test_basic_file_structure),
        ("Test Data", verify_test_data), 
        ("Preprocessing Setup", test_preprocessing_setup)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            logger.error(f"❌ {test_name} failed: {e}")
            results[test_name] = False
    
    # Summary
    passed = sum(results.values())
    total = len(results)
    
    logger.info("\\n" + "=" * 60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 60)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        logger.info(f"{emoji} {test_name}: {status}")
    
    logger.info(f"\\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("\\n🎉 VERIFICATION COMPLETE!")
        logger.info("Test data and structure are properly set up.")
        logger.info("\\nTo run full tests:")
        logger.info("1. Set up virtual environment")
        logger.info("2. Install dependencies: pip install -r requirements.txt")
        logger.info("3. Run: python test_pipeline.py")
    else:
        logger.error("\\n❌ Some verification tests failed")
        logger.error("Check the output above for details")
        
    return 0 if passed == total else 1

if __name__ == "__main__":
    sys.exit(main())