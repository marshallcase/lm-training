from setuptools import setup, find_packages

# Core requirements needed for base functionality
CORE_REQUIREMENTS = [
    "torch>=2.0.0",
    "transformers>=4.28.0",
    "datasets>=2.12.0",
    "tokenizers>=0.13.0",
    "sentencepiece>=0.1.97",
    "numpy>=1.24.0",
    "pandas>=1.5.0",
    "scikit-learn>=1.2.0",
    "tqdm>=4.65.0",
    "tensorboard>=2.12.0",
    "matplotlib>=3.7.0",
    "seaborn>=0.12.0",
    "evaluate>=0.4.0",
    "accelerate>=0.18.0",
    "jsonlines>=3.1.0",
    "pyyaml>=6.0",
]

# Development requirements for testing and code quality
DEV_REQUIREMENTS = [
    "pytest>=7.3.1",
    "black>=23.3.0",
    "isort>=5.12.0",
    "flake8>=6.0.0",
    "jupyter>=1.0.0",
]

# Optimization requirements for high-performance training
OPTIM_REQUIREMENTS = [
    "flash-attn>=1.0.0",
    "bitsandbytes>=0.38.0",
    "triton>=2.0.0",
]

# NLP specific tools that might not be needed for all users
NLP_REQUIREMENTS = [
    "nltk>=3.8.0",
]

# Tracking and visualization
TRACKING_REQUIREMENTS = [
    "wandb>=0.15.0",
    "mlflow>=2.3.0",
]

setup(
    name="lm-training",
    version="0.1.0",
    packages=find_packages(),
    install_requires=CORE_REQUIREMENTS,
    extras_require={
        "dev": DEV_REQUIREMENTS,
        "optim": OPTIM_REQUIREMENTS,
        "nlp": NLP_REQUIREMENTS,
        "tracking": TRACKING_REQUIREMENTS,
        "all": DEV_REQUIREMENTS + OPTIM_REQUIREMENTS + NLP_REQUIREMENTS + TRACKING_REQUIREMENTS,
    },
    python_requires=">=3.9",
    author="Marshall Case",
    author_email="marshall.acecase@gmail.com",
    description="A package for language model training",
    keywords="nlp, transformers, language-models",
    url="https://github.com/marshallcase/lm-training",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
    ],
)