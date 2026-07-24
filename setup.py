from setuptools import setup, find_packages

setup(
    name="zahidgpt",
    version="0.1.0",
    description="Modular Transformer & Multi-Corpus (English, Arabic, Code) LLM Library",
    author="Zahid",
    url="https://github.com/Zahid-coder-17/transformers",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
