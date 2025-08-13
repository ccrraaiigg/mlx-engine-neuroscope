# Project Structure

## Core Engine (`mlx_engine/`)
- **`__init__.py`**: Main exports and initialization
- **`generate.py`**: Text generation with activation capture
- **`api_server.py`**: REST API server with NeuroScope endpoints
- **`activation_hooks.py`**: Core activation capture infrastructure
- **`logging.py`**: Centralized logging configuration

### Model Kits
- **`model_kit/`**: Text model loading and management
  - `model_kit.py`: Core ModelKit with activation hook support
  - `moe_loader.py`, `moe_model.py`: Mixture of Experts support
  - `patches/`: Model-specific patches (Ernie, Gemma)
- **`vision_model_kit/`**: Vision-language model support
- **`external/`**: External model implementations and datasets

### Utilities
- **`utils/`**: Helper modules
  - `register_models.py`: Model registration system
  - `image_utils.py`: Image processing utilities
  - `prompt_processing.py`: Text preprocessing
- **`processors/`**: Text processing components

## Application Code (`code/`)
- **`neuroscope_integration.py`**: NeuroScope bridge and client
- **`start_api_server.py`**: API server launcher
- **`gpt_oss_model.py`**: GPT-OSS model implementation
- **`test_*.py`**: Various test scripts

## Demo System (`demo/`)
- **`run_neuroscope_demo.py`**: Main demo runner with versioning
- **`demo_neuroscope_rest_interface.py`**: REST API demonstration
- **`data/`**: Generated activation data with format documentation
- **`logs/`**: Execution logs organized by version

## Testing (`tests/`)
- **`conftest.py`**: Pytest configuration with heavy test markers
- **`test_*.py`**: Unit tests for core functionality
- **`data/`**: Test data files
- **`processors/`**: Test-specific processors

## Examples (`examples/`)
- **`neuroscope_integration_example.py`**: Comprehensive usage examples

## Models (`models/`)
- Local model storage (gitignored, large files)
- Organized by provider/model structure

## Configuration Files
- **`requirements.txt`**: Python dependencies
- **`.pre-commit-config.yaml`**: Code quality hooks
- **`run_with_limits.sh`**: Resource-limited execution script

## Documentation
- **`README.md`**: Main NeuroScope integration documentation
- **`IMPLEMENTATION_SUMMARY.md`**: Detailed implementation overview
- **`README_*.md`**: Specialized documentation files

## Naming Conventions
- **Snake_case**: Python files and functions
- **PascalCase**: Classes and types
- **UPPER_CASE**: Constants and environment variables
- **Descriptive names**: Clear purpose indication (e.g., `activation_hooks.py`)

## Import Patterns
- **Relative imports**: Within mlx_engine package
- **Absolute imports**: For external dependencies
- **Lazy imports**: For optional dependencies (Flask, FastAPI)
- **Error handling**: Graceful degradation for missing imports

## File Organization Principles
- **Separation of concerns**: Core engine vs application code
- **Modular design**: Independent, testable components
- **Clear boundaries**: API, engine, integration layers
- **Version control**: Generated data and logs organized by timestamp