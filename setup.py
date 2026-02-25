from setuptools import setup, find_packages

setup(
    name="muflon",
    version="1.1.8",
    description="Matrix Utility for Fuzzy Logic Operations and Norms",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Jakub Kiryła",
    author_email="kub-kir@wp.pl",
    url="https://github.com/Kiryl24/muflon",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
)