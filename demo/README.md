# NeuroScope REST Interface Demo

This demo shows how NeuroScope will interact with the MLX Engine REST API for mechanistic interpretability analysis. It elaborates on `test_gpt_oss_20b.py` by demonstrating the complete REST interface workflow.

## Overview

The demo consists of several components that demonstrate the complete workflow for NeuroScope integration:

1. **`demo_neuroscope_rest_interface.py`** - Main demo showing REST API usage
2. **`run_neuroscope_demo.py`** - Simple runner script to execute all demos
3. **`../code/neuroscope_api_reference.py`** - Comprehensive API reference and examples
4. **`data/`** - Generated activation data and format documentation

## What's Demonstrated

### 1. Basic REST Interface
- **Health Checks**: Verify API server is running
- **Model Loading**: Load models via REST API
- **Standard Generation**: Basic chat completions
- **Activation Capture**: Generate text while capturing neural activations
- **Circuit Analysis**: Comprehensive analysis across multiple layers
- **Streaming Concept**: Real-time activation capture during generation

### 2. Integration Workflow
The demo shows the complete NeuroScope integration workflow:
```
NeuroScope → REST API → MLX Engine → Model → Activations → Analysis
```

## Prerequisites

1. **Model**: The gpt-oss-20b model must be available at:
   ```
   ../models/nightmedia/gpt-oss-20b-q4-hi-mlx/
   ```

2. **Dependencies**: Install required Python packages:
   ```bash
   pip install flask flask-cors mlx transformers
   ```

3. **Memory**: Ensure sufficient RAM (32GB+ recommended for full model)

## Running the Demo

### Quick Start

From the project root directory:

```bash
python demo/run_neuroscope_demo.py
```

This will:
1. Check requirements
2. Run the NeuroScope REST interface demo (includes all basic test validation)
3. Display the API reference

**Note**: The comprehensive demo includes the same validation as `test_gpt_oss_20b.py` but via REST API, so running both would be redundant.

### Individual Components

Run specific parts of the demo:

```bash
# Basic functionality test (direct MLX engine testing)
python code/test_gpt_oss_20b.py

# Full NeuroScope REST demo
python demo/demo_neuroscope_rest_interface.py

# API reference and examples
python code/neuroscope_api_reference.py
```

## Key Features Demonstrated

### 1. Activation Capture

The demo shows how to capture neural activations during text generation:

```python
activation_hooks = [
    {
        'layer_name': 'transformer.h.5',
        'component': 'residual',
        'hook_id': 'layer_5_residual',
        'capture_output': True
    }
]

result = client.chat_completion_with_activations(
    messages=messages,
    activation_hooks=activation_hooks
)

# Access captured activations
activations = result['activations']
```

### 2. Circuit Analysis

Multiple analysis scenarios are demonstrated:
- **Mathematical Reasoning**: Arithmetic computation circuits
- **Factual Recall**: Knowledge retrieval mechanisms  
- **Creative Writing**: Generative and creative circuits
- **Attention Patterns**: Multi-head attention analysis
- **Residual Stream**: Information flow tracking

## API Endpoints

### Core Endpoints
- `GET /health` - Health check
- `POST /v1/models/load` - Load model
- `GET /v1/models` - List models
- `POST /v1/chat/completions` - Standard generation
- `POST /v1/chat/completions/with_activations` - Generation with activation capture

### Activation Management
- `POST /v1/activations/hooks` - Register activation hooks
- `DELETE /v1/activations/hooks` - Clear activation hooks

## Generated Data Files

The demo automatically generates:

### Activation Capture
- `data/activation_capture_demo.json` - Raw activation data
- `data/activation_capture_demo_format.md` - Format documentation

### Circuit Analysis
- `data/circuit_analysis_attention_patterns.json` - Attention analysis data
- `data/circuit_analysis_attention_patterns_format.md` - Attention format docs
- `data/circuit_analysis_mlp_processing.json` - MLP analysis data
- `data/circuit_analysis_mlp_processing_format.md` - MLP format docs

## Test Architecture

### Two Complementary Testing Approaches

**`../code/test_gpt_oss_20b.py` - Direct MLX Engine Testing**
- **Purpose**: Validates core MLX engine functionality without REST overhead
- **Use Cases**: 
  - Debugging MLX engine issues in isolation
  - Quick validation of model loading and generation
  - Standalone testing via `run_with_limits.sh`
  - Baseline performance measurement
- **When to use**: When you need to isolate MLX engine issues from REST API issues

**NeuroScope Demo - REST API Integration Testing**
- **Purpose**: Validates the complete NeuroScope integration workflow
- **Use Cases**:
  - End-to-end NeuroScope functionality testing
  - REST API validation
  - Activation capture testing
  - Production workflow validation
- **When to use**: For comprehensive integration testing and NeuroScope development

**Key Insight**: The NeuroScope Demo includes all the validation of the basic test, so running both would be redundant for integration testing. However, the basic test remains valuable for isolated MLX engine debugging and lightweight validation.

## Troubleshooting

### Common Issues
1. **Model Not Found**: Ensure model is downloaded and path is correct
2. **Memory Issues**: Reduce model size or increase system RAM
3. **Port Conflicts**: Change API server port if 8080 is in use
4. **Dependencies**: Install all required packages with pip

### Testing Strategy
- **Start with**: `python demo/run_neuroscope_demo.py` for comprehensive testing
- **Debug with**: `python code/test_gpt_oss_20b.py` if you suspect MLX engine issues
- **Isolate issues**: Use the basic test to separate MLX problems from REST API problems

## Integration with NeuroScope

The generated data files provide concrete examples of:
- Real activation tensor formats
- Multi-layer circuit analysis results
- Temporal dynamics during text generation
- Component isolation (attention vs MLP)
- Complete NeuroScope integration workflow

These files serve as reference implementations for NeuroScope integration development.

## Next Steps

After running the demo:
1. **Implement NeuroScope Client**: Use provided templates
2. **Add Visualization**: Create activation visualization components
3. **Build Analysis Tools**: Implement circuit analysis algorithms
4. **Test Real Workflows**: Try actual mechanistic interpretability tasks