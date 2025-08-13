# NeuroScope Integration for MLX Engine

This document describes the extensions to MLX Engine that enable mechanistic interpretability analysis through integration with NeuroScope, a Smalltalk-based interpretability framework.

## Overview

The extensions provide:

1. **Activation Hook Infrastructure** - Capture internal model activations during inference
2. **Extended REST API** - New endpoints for activation-aware generation
3. **NeuroScope Bridge** - Convert between MLX Engine and NeuroScope data formats
4. **Smalltalk Integration** - Generated code for seamless NeuroScope integration

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   NeuroScope    │    │   REST API       │    │   MLX Engine    │
│   (Smalltalk)   │◄──►│   Server         │◄──►│   + Hooks       │
│                 │    │                  │    │                 │
│ - Circuit Finder│    │ - /v1/chat/...   │    │ - Model Kit     │
│ - Visualizers   │    │ - /activations   │    │ - Hook Manager  │
│ - Analysis Tools│    │ - Bridge Layer   │    │ - Capture Logic │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## Installation

1. **Install Dependencies**:
   ```bash
   pip install flask flask-cors requests numpy
   ```

2. **Verify MLX Engine Installation**:
   ```python
   import mlx_engine
   print("MLX Engine available")
   ```

## Quick Start

### 1. Start the API Server

```bash
# Start the extended API server
python -m mlx_engine.api_server

# Or run directly
python mlx_engine/api_server.py
```

The server will start on `http://127.0.0.1:8080` by default.

### 2. Load a Model

```python
from neuroscope_integration import LMStudioNeuroScopeClient

client = LMStudioNeuroScopeClient()

# Load your model
result = client.load_model("/path/to/your/model")
print(f"Model loaded: {result}")
```

### 3. Generate with Activation Capture

```python
# Define activation hooks
hooks = [
    {
        'layer_name': 'transformer.h.5',
        'component': 'residual',
        'hook_id': 'layer_5_residual'
    },
    {
        'layer_name': 'transformer.h.10',
        'component': 'attention',
        'hook_id': 'layer_10_attention'
    }
]

# Generate with activations
messages = [{"role": "user", "content": "Explain transformers"}]
result = client.generate_with_activations(
    messages=messages,
    activation_hooks=hooks,
    max_tokens=100
)

print(f"Generated: {result['choices'][0]['message']['content']}")
print(f"Captured activations: {len(result['activations'])} hooks")
```

## API Endpoints

### Standard Endpoints

- `GET /health` - Health check
- `GET /v1/models` - List loaded models
- `POST /v1/models/load` - Load a model
- `POST /v1/chat/completions` - Standard chat completions

### Extended Endpoints

- `POST /v1/chat/completions/with_activations` - Generate with activation capture
- `POST /v1/activations/hooks` - Register activation hooks
- `DELETE /v1/activations/hooks` - Clear activation hooks

### Activation Capture Request Format

```json
{
  "messages": [
    {"role": "user", "content": "Your prompt here"}
  ],
  "activation_hooks": [
    {
      "layer_name": "transformer.h.5",
      "component": "residual",
      "hook_id": "layer_5_residual",
      "capture_input": false,
      "capture_output": true
    }
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
```

### Response Format

```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Generated text..."
      },
      "finish_reason": "stop"
    }
  ],
  "activations": {
    "layer_5_residual": [
      {
        "hook_id": "layer_5_residual",
        "layer_name": "transformer.h.5",
        "component": "residual",
        "shape": [1, 10, 768],
        "dtype": "float32",
        "is_input": false,
        "data": [[...]]
      }
    ]
  }
}
```

## Component Types

The following component types can be hooked:

- `residual` - Residual stream activations
- `attention` - Attention layer outputs
- `mlp` - MLP layer outputs
- `embedding` - Embedding layer outputs
- `layernorm` - Layer normalization outputs
- `attention_scores` - Raw attention scores
- `attention_pattern` - Attention patterns
- `key` - Attention key vectors
- `query` - Attention query vectors
- `value` - Attention value vectors

## NeuroScope Integration

### Generate Smalltalk Interface

```python
from neuroscope_integration import generate_neuroscope_smalltalk_interface, LMStudioNeuroScopeClient

client = LMStudioNeuroScopeClient()
smalltalk_code = generate_neuroscope_smalltalk_interface(client)

# Save to file for loading into NeuroScope
with open('LMStudioRESTClient.st', 'w') as f:
    f.write(smalltalk_code)
```

### Use in NeuroScope

```smalltalk
"Load the generated client class into your Smalltalk environment"

"Create a client instance"
client := LMStudioRESTClient default.

"Load a model"
client loadModel: '/path/to/your/model'.

"Create hooks for circuit analysis"
hooks := client createCircuitAnalysisHooks: 24.

"Generate with activation capture"
messages := Array with: (Dictionary new 
    at: #role put: 'user';
    at: #content put: 'Analyze this text';
    yourself).

result := client 
    generateWithActivations: messages 
    hooks: hooks 
    maxTokens: 100 
    temperature: 0.7.

"Convert activations for NeuroScope analysis"
neuroScopeActivations := client convertActivationsForNeuroScope: (result at: #activations).

"Now use NeuroScope's analysis tools"
analyzer := CircuitFinder new.
circuits := analyzer findCircuits: neuroScopeActivations.
```

## Advanced Usage

### Circuit Discovery

```python
from neuroscope_integration import NeuroScopeActivationBridge

bridge = NeuroScopeActivationBridge()

# Create comprehensive hooks for circuit analysis
hooks = bridge.create_hook_specs_for_circuit_analysis(
    model_layers=24,
    components=['residual', 'attention', 'mlp']
)

# Generate with circuit analysis hooks
result = client.generate_with_activations(
    messages=[{"role": "user", "content": "The cat sat on the mat"}],
    activation_hooks=hooks,
    max_tokens=50
)

# Analyze the captured activations
activations = result['activations']
print(f"Captured data from {len(activations)} hooks")
```

### Attention Pattern Analysis

```python
# Create hooks specifically for attention analysis
attention_hooks = bridge.create_attention_analysis_hooks(
    model_layers=24,
    target_layers=[5, 10, 15, 20]  # Focus on specific layers
)

result = client.generate_with_activations(
    messages=[{"role": "user", "content": "John gave Mary the book because she asked for it"}],
    activation_hooks=attention_hooks,
    max_tokens=1  # Just process the prompt
)

# Analyze attention patterns for pronoun resolution
attention_data = result['activations']
# Process with NeuroScope attention analyzers...
```

### Streaming with Activations

```python
# Enable streaming for real-time analysis
result = client.generate_with_activations(
    messages=[{"role": "user", "content": "Write a story"}],
    activation_hooks=hooks,
    stream=True,
    max_tokens=200
)

# Process streaming results with activations
# (Implementation depends on your specific needs)
```

## Performance Considerations

1. **Memory Usage**: Activations can be large. Consider:
   - Limiting the number of hooks
   - Using specific layers rather than all layers
   - Clearing activations regularly

2. **Computation Overhead**: Hooks add processing time:
   - Start with fewer hooks for initial analysis
   - Use `capture_output=True, capture_input=False` unless you need both

3. **Network Transfer**: For remote analysis:
   - Consider compression for large activation tensors
   - Use streaming for long generations

## Troubleshooting

### Common Issues

1. **"Model does not support activation hooks"**
   - Ensure you're using ModelKit (not VisionModelKit for text-only models)
   - Check that the model loaded successfully

2. **"Could not find module for layer"**
   - Verify layer names match your model architecture
   - Use model inspection to find correct layer names

3. **Memory errors with large models**
   - Reduce the number of simultaneous hooks
   - Use smaller batch sizes
   - Consider quantization options

### Debugging

```python
# Check model architecture
model = load_model("/path/to/model")
print(f"Model type: {type(model)}")
print(f"Has activation hooks: {hasattr(model, 'activation_hook_manager')}")

# Test basic hook registration
try:
    hook_id = model.register_activation_hook(
        layer_name='transformer.h.0',
        component='residual'
    )
    print(f"Successfully registered hook: {hook_id}")
    model.unregister_activation_hook(hook_id)
except Exception as e:
    print(f"Hook registration failed: {e}")
```

## Examples

See `examples/neuroscope_integration_example.py` for comprehensive examples including:

- Basic activation capture
- API server usage
- NeuroScope bridge conversion
- Circuit analysis setup
- Attention pattern analysis

## Contributing

To extend the integration:

1. **Add new component types** in `activation_hooks.py`
2. **Extend the API** in `api_server.py`
3. **Update the bridge** in `neuroscope_integration.py`
4. **Generate new Smalltalk interfaces** as needed

## License

This integration maintains the same license as the base MLX Engine project.