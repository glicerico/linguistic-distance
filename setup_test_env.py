#!/usr/bin/env python3
"""
Setup test environment with minimal dependencies for testing the pipeline.
This script creates a minimal test environment to verify core functionality.
"""

import sys
import os
from pathlib import Path
import shutil


def create_minimal_test_data():
    """Create minimal test data files."""
    print("Creating minimal test data...")
    
    # Ensure test directory exists
    test_dir = Path("data/test")
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # Create minimal test files if they don't exist
    test_data = {
        "english": [
            "the cat sat on the mat",
            "the dog ran in the park", 
            "birds fly in the sky",
            "fish swim in the water",
            "children play with toys"
        ],
        "spanish": [
            "el gato se sentó en la alfombra",
            "el perro corrió en el parque",
            "los pájaros vuelan en el cielo", 
            "los peces nadan en el agua",
            "los niños juegan con juguetes"
        ],
        "german": [
            "die katze saß auf der matte",
            "der hund lief im park",
            "vögel fliegen am himmel",
            "fische schwimmen im wasser",
            "kinder spielen mit spielzeug"
        ]
    }
    
    for lang, sentences in test_data.items():
        test_file = test_dir / f"{lang}_test.txt"
        with open(test_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sentences))
        print(f"  Created {test_file}")
    
    return test_data


def check_python_packages():
    """Check if required packages are available."""
    print("Checking Python packages...")
    
    required_packages = [
        'numpy', 'scipy', 'scikit-learn', 'pandas',
        'matplotlib', 'seaborn', 'tqdm', 'gensim'
    ]
    
    missing_packages = []
    available_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            available_packages.append(package)
            print(f"  ✅ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"  ❌ {package}")
    
    return available_packages, missing_packages


def create_venv_instructions():
    """Create instructions for setting up virtual environment."""
    instructions = """
# VIRTUAL ENVIRONMENT SETUP INSTRUCTIONS

## Option 1: Using venv (Python built-in)
```bash
# Create virtual environment
python3 -m venv linguistic_distance_venv

# Activate virtual environment
# On Linux/Mac:
source linguistic_distance_venv/bin/activate
# On Windows:
# linguistic_distance_venv\\Scripts\\activate

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_pipeline.py
```

## Option 2: Using conda
```bash
# Create conda environment
conda create -n linguistic_distance python=3.8

# Activate environment
conda activate linguistic_distance

# Install dependencies
pip install -r requirements.txt

# Run tests
python test_pipeline.py
```

## Quick Test (after activation)
```bash
# Test basic imports
python -c "import numpy, scipy, gensim; print('Core packages available!')"

# Run pipeline test
python test_pipeline.py
```
"""
    
    with open("VENV_SETUP.md", 'w') as f:
        f.write(instructions)
    
    print("Created VENV_SETUP.md with virtual environment instructions")


def main():
    """Set up test environment."""
    print("=" * 60)
    print("LINGUISTIC DISTANCE - TEST ENVIRONMENT SETUP")
    print("=" * 60)
    
    # Create test data
    test_data = create_minimal_test_data()
    print(f"✅ Created test data for {len(test_data)} languages")
    
    # Check available packages
    available, missing = check_python_packages()
    
    print(f"\n📦 Package Status:")
    print(f"  Available: {len(available)}/{len(available + missing)}")
    print(f"  Missing: {missing}")
    
    # Create setup instructions
    create_venv_instructions()
    
    if missing:
        print("\n⚠️  Missing packages detected!")
        print("Please set up a virtual environment and install dependencies:")
        print("\n1. Create and activate virtual environment (see VENV_SETUP.md)")
        print("2. Install dependencies: pip install -r requirements.txt")
        print("3. Run test: python test_pipeline.py")
    else:
        print("\n✅ All packages available! You can run:")
        print("python test_pipeline.py")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()