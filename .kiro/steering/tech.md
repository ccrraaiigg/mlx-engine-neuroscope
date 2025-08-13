# Technology Stack

## Core Framework
- **MLX**: Apple's machine learning framework for Apple Silicon
- **Python 3.13**: Primary development language
- **PyTorch**: For model compatibility and transformations
- **Transformers**: Hugging Face library for model loading

## Key Dependencies
- **mlx-lm**: MLX language model utilities
- **mlx-vlm**: MLX vision-language models (custom branch)
- **Flask + Flask-CORS**: REST API server
- **FastAPI**: Alternative API framework
- **NumPy**: Numerical computations
- **Safetensors**: Model weight storage format

## Development Tools
- **pytest**: Testing framework with heavy test markers
- **pre-commit**: Code quality hooks
- **setuptools**: Package management

## Build Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv_test_gpt
source .venv_test_gpt/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Testing
```bash
# Run standard tests
pytest

# Run heavy tests (requires large models)
pytest --heavy

# Quick functionality test
python code/test_optimized.py

# Comprehensive model test
python code/test_gpt_oss_20b.py
```

### API Server
```bash
# Start API server (default port 8080)
python -m mlx_engine.api_server

# Start on custom port
python code/start_api_server.py  # Port 50111
```

### Demo System
```bash
# Run complete NeuroScope demo
python demo/run_neuroscope_demo.py

# REST interface demo
python demo/demo_neuroscope_rest_interface.py
```

## Architecture Patterns
- **Hook-based activation capture**: Non-intrusive model instrumentation
- **REST API with streaming**: OpenAI-compatible endpoints
- **Modular design**: Separate concerns (hooks, API, bridge)
- **Memory-efficient**: Configurable activation retention
- **Error-resilient**: Graceful degradation and cleanup

## Platform Requirements
- **macOS**: Primary target (Apple Silicon preferred)
- **Memory**: 28GB+ recommended for large models
- **Storage**: Models can be 10GB+ (quantized versions available)