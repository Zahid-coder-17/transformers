from setuptools import setup, find_packages

setup(
    name="zahidgpt",
    version="0.1.0",
    description="Modular Transformer & Multi-Corpus (English, Arabic, Code) LLM Library",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Zahid",
    author_email="zahid.coder.17@gmail.com",
    url="https://github.com/Zahid-coder-17/transformers",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "zahidgpt": ["checkpoints/*", "data/*"],
    },
    install_requires=[
        "torch>=2.0.0",
        "huggingface_hub",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
