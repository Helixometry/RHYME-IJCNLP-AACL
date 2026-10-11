from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="RHYME",
    version="1.0.0",
    author="Farhan Sheth, Girish, Mohd Mujtaba Akhtar, Muskaan Singh",
    author_email="girish281003@gmail.com",
    description="Generalizing Speech Deepfake Detection using Non-Euclidean Geometry",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/farhan-sheth/RHYME-IJCNLP-AACL",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.9.0",
        "torchaudio>=0.9.0",
        "librosa>=0.9.0",
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "geoopt>=0.2.0",
        "geomstats>=0.50.0",
        "PyYAML>=5.3",
        "pandas>=1.1.0",
        "scikit-learn>=0.23.0",
        "tqdm>=4.50.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
)
