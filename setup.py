from setuptools import setup, find_packages

setup(
    name="ai-photo-editor",
    version="0.1.0",
    description="AI-powered RAW photo editor using Ollama",
    author="User",
    packages=find_packages(),
    install_requires=[
        "requests>=2.28.0",
        "Pillow>=9.0.0",
        "rawpy>=0.15.0",
        "numpy>=1.21.0",
        "imageio>=2.25.0",
    ],
    entry_points={
        "console_scripts": [
            "ai-photo-editor=ai_photo_editor:main",
        ],
    },
    python_requires=">=3.7",
)
