# Activation Capture Data Format

Generated: 2025-08-07T23:14:49.356007
Demo Type: Basic Activation Capture

## File Structure

```json
{
  "timestamp": "ISO 8601 timestamp of capture",
  "demo_type": "activation_capture", 
  "total_hooks": 2,
  "activations": {
    "hook_id": [
      {
        "hook_id": "string - unique identifier for this hook",
        "layer_name": "string - model layer name (e.g., 'model.layers.0.self_attn')",
        "component": "string - component type ('attention', 'mlp', 'residual', etc.)",
        "shape": [batch_size, sequence_length, hidden_size],
        "dtype": "string - data type ('float32', 'float16', etc.)",
        "is_input": boolean - whether this is input or output activation
      }
    ]
  }
}
```

## Hook Details

### Hook: attention_layer_0
- **Layer**: model.layers.0.self_attn
- **Component**: attention
- **Activations Count**: 20
- **Shape**: [1, 32, 768]
- **Data Type**: float32
- **Is Input**: False

### Hook: mlp_layer_5
- **Layer**: model.layers.5.mlp
- **Component**: mlp
- **Activations Count**: 20
- **Shape**: [1, 32, 768]
- **Data Type**: float32
- **Is Input**: False

## Usage for NeuroScope

This data format is designed for mechanistic interpretability analysis:

1. **Hook ID**: Use to identify which layer/component the activation came from
2. **Shape**: [batch_size, sequence_length, hidden_size] tensor dimensions
3. **Layer Name**: Maps to specific transformer architecture components
4. **Component Type**: Distinguishes between attention, MLP, residual connections
5. **Activation Count**: Number of tokens/steps captured during generation

## Integration Notes

- Each activation represents one forward pass step during text generation
- Multiple activations per hook indicate multi-token generation
- Shape [1, 32, 768] indicates: 1 batch, 32 sequence length, 768 hidden dimensions
- Data can be loaded and processed by NeuroScope for circuit analysis
