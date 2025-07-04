from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="credit-default-prediction",
    version="1.0.0",
    author="MOHD AFROZ ALI",
    author_email="afrozali3001.aa@gmail.com",
    description="End-to-End Credit Default Prediction with Explainable AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MOHD-AFROZ-ALI/Credit_Default_Predict",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
            "pre-commit>=2.17.0",
        ],
        "docs": [
            "sphinx>=4.0.0",
            "sphinx-rtd-theme>=1.0.0",
            "myst-parser>=0.17.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "credit-default-train=credit_default.pipeline.training_pipeline:main",
            "credit-default-predict=credit_default.pipeline.prediction_pipeline:main",
        ],
    },
    include_package_data=True,
    package_data={
        "credit_default": ["config/*.yaml", "data/*.csv"],
    },
)