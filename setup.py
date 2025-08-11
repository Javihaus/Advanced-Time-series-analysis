"""
Setup script for the Advanced Time Series Analysis Framework.
"""

from setuptools import setup, find_packages

# Read the README file for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="advanced-time-series-analysis",
    version="1.0.0",
    author="Javier Marin",
    author_email="javier@jmarin.info",
    description="A comprehensive framework for comparing probabilistic programming, deep learning, and gradient boosting methods for time series forecasting",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Javihaus/Advanced-Time-series-analysis",
    project_urls={
        "Bug Tracker": "https://github.com/Javihaus/Advanced-Time-series-analysis/issues",
        "Documentation": "https://github.com/Javihaus/Advanced-Time-series-analysis",
        "Source Code": "https://github.com/Javihaus/Advanced-Time-series-analysis",
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "matplotlib>=3.5.0",
    ],
    extras_require={
        "deep-learning": [
            "torch>=1.11.0",
            "tensorflow>=2.8.0",
        ],
        "probabilistic": [
            "pymc3>=3.11.0",
            "theano-pymc>=1.1.0",
            "arviz>=0.11.0",
        ],
        "boosting": [
            "xgboost>=1.5.0",
        ],
        "visualization": [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        "statistical": [
            "statsmodels>=0.13.0",
        ],
        "dev": [
            "jupyter>=1.0.0",
            "ipykernel>=6.0.0",
            "pytest>=6.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.910",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
        ],
        "all": [
            "torch>=1.11.0",
            "tensorflow>=2.8.0",
            "pymc3>=3.11.0",
            "theano-pymc>=1.1.0",
            "arviz>=0.11.0",
            "xgboost>=1.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "statsmodels>=0.13.0",
        ]
    },
    include_package_data=True,
    keywords=[
        "time-series",
        "forecasting", 
        "machine-learning",
        "deep-learning",
        "gaussian-processes",
        "lstm",
        "rnn",
        "gru",
        "xgboost",
        "bayesian-inference",
        "probabilistic-programming",
        "model-comparison"
    ],
    zip_safe=False,
)