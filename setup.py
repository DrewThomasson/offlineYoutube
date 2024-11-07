import sys
import platform
from setuptools import setup, find_packages

# Check for M1 Mac and Python version
if platform.system() == "Darwin" and platform.processor() == "arm":
    if not (sys.version_info.major == 3 and sys.version_info.minor == 10):
        raise RuntimeError(
            "This package requires Python 3.10 on M1 Macs. "
            "Please create a Python 3.10 virtual environment and try again."
        )

setup(
    name="offlineyoutube",
    version="2.2.0b",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "yt-dlp",
        "pandas",
        "numpy",
        "requests",
        "faiss-cpu",
        "faster-whisper",
        "sentence-transformers",
        "gradio==3.36.1",
        "argparse",
        "beautifulsoup4",
        "pysrt",
        "webvtt-py"
    ],
    entry_points={
        "console_scripts": [
            "offlineyoutube=offlineyoutube.app:main"
        ]
    },
    python_requires=">=3.8",
    author="Andrew Phillip Thomasson",
    author_email="drew.thomasson100@gmail.com",
    description="A YouTube video search and management tool with a Gradio interface",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DrewThomasson/offlineYoutube",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
