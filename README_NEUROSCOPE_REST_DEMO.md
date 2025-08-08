# NeuroScope REST Interface Demo

This demo elaborates on `test_gpt_oss_20b.py` by showing how NeuroScope will interact with the MLX Engine REST API for mechanistic interpretability analysis.

## Overview

The demo consists of several components that demonstrate the complete workflow for NeuroScope integration:

1. **`demo/demo_neuroscope_rest_interface.py`** - Main demo showing REST API usage
2. **`neuroscope_api_reference.py`** - Comprehensive API reference and examples
3. **`demo/run_neuroscope_demo.py`** - Simple runner script to execute all demos
4. **This README** - Documentation and usage instructions

## What's Demonstrated

### 1. Basic REST Interface (`demo/demo_neuroscope_rest_interface.py`)

- **Health Checks**: Verify API server is running
- **Model Loading**: Load models via REST API
- **Standard Generation**: Basic chat completions
- **Activation Capture**: Generate text while capturing neural activations
- **Circuit Analysis**: Comprehensive analysis across multiple layers
- **Streaming Concept**: Real-time activation capture during generation

### 2. API Reference (`neuroscope_api_reference.py`)

- **Complete Endpoint Documentation**: All REST endpoints with examples
- **Data Structures**: Request/response formats for NeuroScope
- **Hook Configurations**: Pre-built activation hook setups
- **Test Scenarios**: Common analysis scenarios for mechanistic interpretability
- **Client Templates**: JavaScript/TypeScript client implementation examples

### 3. Integration Workflow

The demo shows the complete NeuroScope integration workflow:

```
NeuroScope → REST API → MLX Engine → Model → Activations → Analysis
```

## Prerequisites

1. **Model**: The gpt-oss-20b model must be available at:
   ```
   ./models/nightmedia/gpt-oss-20b-q4-hi-mlx/
   ```

2. **Dependencies**: Install required Python packages:
   ```bash
   pip install flask flask-cors mlx transformers
   ```

3. **Memory**: Ensure sufficient RAM (32GB+ recommended for full model)

## Running the Demo

### Quick Start

Run all demos with the simple runner:

```bash
python demo/run_neuroscope_demo.py
```

This will:
1. Check requirements
2. Run the basic test (from `test_gpt_oss_20b.py`)
3. Run the NeuroScope REST interface demo
4. Display the API reference

### Individual Components

Run specific parts of the demo:

```bash
# Basic functionality test
python test_gpt_oss_20b.py

# Full NeuroScope REST demo
python demo/demo_neuroscope_rest_interface.py

# API reference and examples
python neuroscope_api_reference.py
```

### Manual API Server

To run the API server manually for testing:

```bash
python -m mlx_engine.api_server
```

Then interact with it using curl or other HTTP clients:

```bash
# Health check
curl http://127.0.0.1:8080/health

# Load model
curl -X POST http://127.0.0.1:8080/v1/models/load \
  -H "Content-Type: application/json" \
  -d '{"model_path": "./models/nightmedia/gpt-oss-20b-q4-hi-mlx"}'

# Generate with activations
curl -X POST http://127.0.0.1:8080/v1/chat/completions/with_activations \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "activation_hooks": [{
      "layer_name": "transformer.h.5",
      "component": "attention",
      "hook_id": "test_hook"
    }]
  }'
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

### 3. Real-time Analysis

The demo shows concepts for streaming generation with real-time activation capture, enabling live circuit analysis during text generation.

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

## Data Formats

### Activation Hook Structure

```json
{
  "layer_name": "transformer.h.5",
  "component": "attention",
  "hook_id": "unique_hook_id",
  "capture_input": false,
  "capture_output": true
}
```

### Activation Data Structure

```json
{
  "hook_id": "unique_hook_id",
  "layer_name": "transformer.h.5",
  "component": "attention",
  "shape": [1, 10, 768],
  "dtype": "float32",
  "is_input": false,
  "data": "... tensor data ..."
}
```

## NeuroScope Integration

### Client Implementation

The demo provides templates for implementing NeuroScope clients:

1. **JavaScript/TypeScript**: Web-based client for browser integration
2. **Python**: Reference implementation for testing
3. **REST API**: Standard HTTP interface for any language

### Analysis Workflows

Common mechanistic interpretability workflows demonstrated:

1. **Circuit Discovery**: Find computational circuits across layers
2. **Attention Analysis**: Analyze attention patterns and heads
3. **Information Flow**: Track information through residual stream
4. **Component Interaction**: Study attention-MLP interactions

## Comparison with Basic Test

The demo elaborates on `test_gpt_oss_20b.py` by adding:

- **REST API Interface**: Full HTTP API instead of direct function calls
- **Activation Capture**: Neural activation recording during generation
- **Multiple Scenarios**: Various analysis scenarios beyond basic math
- **Comprehensive Hooks**: Multiple activation hooks across layers
- **Real-world Workflow**: Complete NeuroScope integration pattern

## Troubleshooting

### Common Issues

1. **Model Not Found**: Ensure model is downloaded and path is correct
2. **Memory Issues**: Reduce model size or increase system RAM
3. **Port Conflicts**: Change API server port if 8080 is in use
4. **Dependencies**: Install all required packages with pip

### Performance Tips

1. **Selective Hooks**: Use fewer activation hooks for faster generation
2. **Batch Processing**: Process multiple prompts efficiently
3. **Memory Management**: Clear activations between analyses
4. **Streaming**: Use streaming for real-time analysis

## Next Steps

After running the demo:

1. **Implement NeuroScope Client**: Use provided templates
2. **Add Visualization**: Create activation visualization components
3. **Build Analysis Tools**: Implement circuit analysis algorithms
4. **Test Real Workflows**: Try actual mechanistic interpretability tasks

## Files Generated

The demo creates several reference files:

- **API Documentation**: Complete endpoint reference
- **Client Templates**: Implementation examples
- **Hook Configurations**: Pre-built analysis setups
- **Test Scenarios**: Common analysis patterns

These files serve as a foundation for NeuroScope integration development.

## Support

For issues or questions:

1. Check the console output for detailed error messages
2. Verify all prerequisites are met
3. Review the API reference for correct usage patterns
4. Test with the basic `test_gpt_oss_20b.py` first

The demo provides a comprehensive foundation for integrating NeuroScope with MLX Engine for advanced mechanistic interpretability analysis.