from setuptools import setup, find_packages

setup(
    name="speech_module",
    version="0.1.0",
    description="Module professionnel de Speech-to-Text modulaire",
    packages=find_packages(exclude=["tests*", "examples*"]),
    install_requires=[
        "faster-whisper>=1.0.0",
        "sounddevice>=0.4.6",
        "numpy>=1.24.0",
        "torch>=2.0.0",
        "torchaudio>=2.0.0",
        "soundfile>=0.12.1",
        "fastapi>=0.100.0",
        "uvicorn>=0.22.0",
        "python-multipart>=0.0.6",
        "requests>=2.31.0"
    ],
    python_requires=">=3.11",
)
