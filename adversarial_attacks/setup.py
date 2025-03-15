from setuptools import setup, find_packages

setup(
    name="adversarial_attacks", 
    version="0.1.0",  
    author="Erfan Shakouri",  
    author_email="", 
    description="A library for applying adversarial attacks on deep learning models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/adversarial_attacks",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "matplotlib>=3.0.0",
        "numpy>=1.19.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)