from setuptools import setup, find_packages

setup(
    name="linguistic-distance",
    version="0.1.0",
    description="Tools to compare the distance between embedding spaces of monolingual corpora",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "gensim>=4.0.0",
        "fasttext>=0.9.2",
        "nltk>=3.6.0",
        "spacy>=3.4.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.10.0",
        "tqdm>=4.62.0",
        "pot>=0.8.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.2.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "download-data=scripts.download_data:main",
            "train-embeddings=scripts.train_embeddings:main",
            "compute-distances=scripts.compute_distances:main",
            "visualize-results=scripts.visualize_results:main",
        ],
    },
)