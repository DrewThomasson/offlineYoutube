from setuptools import setup, find_packages

setup(
    name="offline_youtube",
    version="0.1",
    packages=find_packages(include=["lib", "lib.*"]),  # Include only the 'lib' package
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
            "offlineYoutube=app:main"  # Directly point to app.py's main function
        ]
    },
)

