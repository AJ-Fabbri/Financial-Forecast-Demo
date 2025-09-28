#!/usr/bin/env python3
"""
Setup script for Financial Forecast Demo
"""
from setuptools import setup, find_packages

setup(
    name="financial-forecast-demo",
    version="1.0.0",
    description="Financial Time Series Prediction Framework Demo",
    author="Anderson Fabbri",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.1.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "scikit-learn>=0.24.0",
        "yfinance>=0.1.63",
        "openpyxl>=3.0.0",
        "statsmodels>=0.12.0",
        "click>=8.0.0",
        "tqdm>=4.60.0",
        "duckdb>=1.0.0",
        "streamlit>=1.0.0",
        "plotly>=5.0.0"
    ],
    python_requires=">=3.9",
    entry_points={
        'console_scripts': [
            'financial-forecast-demo=delphi.scripts.run:main',
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
) 