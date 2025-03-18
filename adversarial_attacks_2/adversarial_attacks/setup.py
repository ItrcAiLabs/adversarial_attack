# setup.py
from setuptools import setup, find_packages

setup(
    name="adversarial_attacks", 
    version="0.1.0",             
    description="A library for adversarial attacks on deep learning models",
    author="Erfan Shakouri",
    packages=find_packages(),    
    install_requires=[           
        "tensorflow>=2.0.0",
        "numpy",
    ],
    python_requires=">=3.6",     
)