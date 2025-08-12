#!/usr/bin/env python3
"""
Basic structure test to verify the repository implementation.
This test checks that all files exist and have proper structure without requiring dependencies.
"""

from pathlib import Path
import ast
import sys


def test_file_structure():
    """Test that all expected files exist."""
    print("=" * 50)
    print("TESTING FILE STRUCTURE")
    print("=" * 50)
    
    required_files = [
        # Core library files
        "src/__init__.py",
        "src/data/__init__.py",
        "src/data/downloader.py",
        "src/data/preprocessor.py",
        "src/embeddings/__init__.py", 
        "src/embeddings/trainer.py",
        "src/embeddings/loader.py",
        "src/alignment/__init__.py",
        "src/alignment/linear_mapping.py",
        "src/alignment/procrustes.py",
        "src/distance/__init__.py",
        "src/distance/earth_movers.py",
        "src/distance/cosine_based.py",
        "src/distance/geometric.py",
        "src/utils/__init__.py",
        "src/utils/visualization.py",
        "src/utils/io.py",
        
        # Scripts
        "scripts/download_data.py",
        "scripts/train_embeddings.py", 
        "scripts/compute_distances.py",
        "scripts/visualize_results.py",
        
        # Examples and tests
        "examples/basic_usage.py",
        "examples/advanced_analysis.py",
        "test_pipeline.py",
        
        # Configuration files
        "requirements.txt",
        "setup.py",
        "pyproject.toml",
        
        # Documentation
        "README.md",
        "CLAUDE.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"✅ Found {len(existing_files)} required files")
    print(f"❌ Missing {len(missing_files)} files")
    
    if missing_files:
        print("Missing files:")
        for file in missing_files:
            print(f"  - {file}")
        return False
    else:
        print("🎉 All required files present!")
        return True


def test_python_syntax():
    """Test that all Python files have valid syntax."""
    print("\\n" + "=" * 50)
    print("TESTING PYTHON SYNTAX")
    print("=" * 50)
    
    python_files = []
    for pattern in ["src/**/*.py", "scripts/*.py", "examples/*.py", "*.py"]:
        python_files.extend(Path(".").glob(pattern))
    
    syntax_errors = []
    valid_files = []
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            ast.parse(content)
            valid_files.append(str(file_path))
        except SyntaxError as e:
            syntax_errors.append((str(file_path), str(e)))
        except Exception as e:
            syntax_errors.append((str(file_path), f"Error reading file: {e}"))
    
    print(f"✅ {len(valid_files)} files have valid syntax")
    print(f"❌ {len(syntax_errors)} files have syntax errors")
    
    if syntax_errors:
        print("Files with syntax errors:")
        for file, error in syntax_errors:
            print(f"  - {file}: {error}")
        return False
    else:
        print("🎉 All Python files have valid syntax!")
        return True


def test_imports():
    """Test that core modules can be imported (without external dependencies)."""
    print("\\n" + "=" * 50)  
    print("TESTING CORE STRUCTURE")
    print("=" * 50)
    
    # Add src to path
    sys.path.insert(0, str(Path("src")))
    
    core_modules = [
        "data",
        "embeddings", 
        "alignment",
        "distance",
        "utils"
    ]
    
    import_errors = []
    successful_imports = []
    
    for module_name in core_modules:
        try:
            module = __import__(module_name)
            successful_imports.append(module_name)
            print(f"✅ {module_name}: OK")
        except Exception as e:
            import_errors.append((module_name, str(e)))
            print(f"❌ {module_name}: {e}")
    
    print(f"\\n✅ {len(successful_imports)} modules imported successfully")
    print(f"❌ {len(import_errors)} modules failed to import")
    
    return len(import_errors) == 0


def test_class_definitions():
    """Test that key classes are properly defined."""
    print("\\n" + "=" * 50)
    print("TESTING CLASS DEFINITIONS") 
    print("=" * 50)
    
    key_files_and_classes = {
        "src/data/downloader.py": ["BibleDownloader"],
        "src/data/preprocessor.py": ["TextPreprocessor"],
        "src/embeddings/trainer.py": ["EmbeddingTrainer"],
        "src/embeddings/loader.py": ["EmbeddingLoader"],
        "src/alignment/linear_mapping.py": ["LinearMapping"],
        "src/alignment/procrustes.py": ["ProcrustesAlignment"],
        "src/distance/earth_movers.py": ["EarthMoversDistance"],
        "src/distance/cosine_based.py": ["CosineSimilarityMetrics"],
        "src/distance/geometric.py": ["GeometricDistances"],
    }
    
    missing_classes = []
    found_classes = []
    
    for file_path, expected_classes in key_files_and_classes.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            # Extract class names from AST
            actual_classes = []
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    actual_classes.append(node.name)
            
            for expected_class in expected_classes:
                if expected_class in actual_classes:
                    found_classes.append(f"{file_path}:{expected_class}")
                    print(f"✅ {expected_class} found in {file_path}")
                else:
                    missing_classes.append(f"{file_path}:{expected_class}")
                    print(f"❌ {expected_class} missing from {file_path}")
                    
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
            for expected_class in expected_classes:
                missing_classes.append(f"{file_path}:{expected_class}")
    
    print(f"\\n✅ {len(found_classes)} classes found")
    print(f"❌ {len(missing_classes)} classes missing")
    
    return len(missing_classes) == 0


def main():
    """Run all structure tests."""
    print("🔍 LINGUISTIC DISTANCE REPOSITORY - STRUCTURE TEST")
    print("=" * 60)
    
    tests = [
        ("File Structure", test_file_structure),
        ("Python Syntax", test_python_syntax),
        ("Core Imports", test_imports),
        ("Class Definitions", test_class_definitions)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        emoji = "✅" if result else "❌"
        print(f"{emoji} {test_name}: {status}")
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL STRUCTURE TESTS PASSED!")
        print("The linguistic distance repository implementation is structurally complete.")
    else:
        print(f"\\n⚠️  {total - passed} tests failed")
        print("Some components may be missing or have issues.")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)