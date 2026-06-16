# AI Photo Editor

An AI-powered professional RAW photo editor that leverages Large Language Models (via Ollama) to analyze images and provide tailored editing recommendations, which are then automatically applied to generate high-quality JPEGs.

## 🌟 Features

- **AI-Driven Adjustments**: Connects to an Ollama server to get professional photography recommendations for exposure, contrast, saturation, and clarity.
- **Wide RAW Support**: Supports a comprehensive list of RAW formats including `.cr2`, `.cr3`, `.nef`, `.orf`, `.arw`, `.dng`, and more via `rawpy`.
- **Immutability**: Adheres to a strict non-destructive workflow. Original RAW files are never modified; all outputs are saved to a separate output directory.
- **Smart Previews**: Automatically generates low-resolution previews to send to the AI, reducing bandwidth and latency while maintaining analysis quality.
- **Detailed Audit Trail**: Every processed image is accompanied by a JSON metadata file containing the exact AI recommendations and the logic used for the edit.
- **Production Ready**: Includes a structured project layout, dependency management, and a comprehensive unit test suite.

## 🚀 Getting Started

### Prerequisites

- **Python 3.7+**
- **Ollama**: An Ollama server running locally or remotely.
- **Model**: A vision-capable model (e.g., `llava`, `bakllava`, or `nemotron3:33b`) that supports image analysis.

### Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd image-editor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

Run the editor via the command line:
```bash
python ai_photo_editor.py /path/to/raw/images --output_dir /path/to/output
```

## 🛠️ How it Works

The pipeline follows these steps:
1. **Analysis**: The tool sends a low-res version of the image to the AI model to evaluate lighting, composition, and color.
2. **Recommendation**: The AI returns a set of suggested adjustments in JSON format (e.g., "Increase exposure by +0.5, decrease contrast").
3. **Execution**: The tool applies these numeric adjustments to the original RAW/high-res image using the processing engine.
4. **Export**: The final image is saved as a high-quality JPEG.

## 📁 Project Structure

- `ai_photo_editor.py`: Main application entry point and core logic.
- `requirements.txt`: List of required Python packages.
- `setup.py`: Packaging configuration for installation.
- `tests/`: Unit tests for validating the image processing and AI response pipelines.

## ⚙️ Configuration

You can configure the AI endpoint and model via environment variables:
- `OLLAMA_SERVER_URL`: The URL of your Ollama server (defaults to `http://localhost:11434`).
- `OLLAMA_MODEL`: The specific model to use (defaults to `nemotron3:33b`).
- `OLLAMA_API_KEY`: Bearer token for authorized Ollama servers.

## 📝 License

This project is licensed under the MIT License.
