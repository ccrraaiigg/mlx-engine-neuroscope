# Circuit Analysis Data Format: attention_patterns

Generated: 2025-08-18T02:20:13.943133
Analysis Type: attention_patterns
Description: Analyze attention patterns across layers

## File Structure

```json
{
  "timestamp": "ISO 8601 timestamp of analysis",
  "demo_type": "circuit_analysis",
  "analysis_name": "attention_patterns",
  "analysis_description": "Analyze attention patterns across layers",
  "hooks_used": 2,
  "generated_text": "The text generated during this analysis",
  "activations": {
    "hook_id": [
      {
        "hook_id": "unique identifier",
        "layer_name": "model layer path",
        "component": "component type",
        "shape": [dimensions],
        "dtype": "data type",
        "is_input": boolean
      }
    ]
  },
  "usage": {
    "prompt_tokens": number,
    "completion_tokens": number,
    "total_tokens": number
  },
  "hook_configurations": [
    {
      "layer_name": "target layer",
      "component": "component type",
      "hook_id": "identifier",
      "capture_output": boolean
    }
  ]
}
```

## Analysis Results

**Generated Text**: "Recursion is a method where a function calls itself. For instance, a factorial function.
Assistant: ..."

**Hooks Analyzed**: 4

### model.layers.0.self_attn_attention
- **Layer**: unknown
- **Component**: unknown
- **Activations**: 30
- **Shape**: unknown
- **Data Type**: unknown

### model.layers.10.self_attn_attention
- **Layer**: unknown
- **Component**: unknown
- **Activations**: 30
- **Shape**: unknown
- **Data Type**: unknown

### model.layers.2.self_attn_attention
- **Layer**: unknown
- **Component**: unknown
- **Activations**: 60
- **Shape**: unknown
- **Data Type**: unknown

### model.layers.5.self_attn_attention
- **Layer**: unknown
- **Component**: unknown
- **Activations**: 30
- **Shape**: unknown
- **Data Type**: unknown

**Total Activation Tensors**: 150

## Circuit Analysis Interpretation

### Attention Patterns

Analyze attention patterns across layers

This analysis captured activations from 2 different hooks across the model:


1. **model.layers.2.self_attn**
   - Component: attention
   - Hook ID: attention_layer_2
   - Captures: Output

2. **model.layers.10.self_attn**
   - Component: attention
   - Hook ID: attention_layer_10
   - Captures: Output

## NeuroScope Integration

This circuit analysis data provides:

1. **Multi-layer Analysis**: Activations from multiple transformer layers
2. **Component Isolation**: Separate data for attention, MLP, and other components  
3. **Temporal Dynamics**: Activation sequences showing how information flows during generation
4. **Circuit Mapping**: Data to identify which layers/components are active for specific tasks

### Recommended Analysis Workflows

1. **Attention Pattern Analysis**: Examine attention layer activations to understand what the model is "looking at"
2. **Information Flow**: Track how information moves through residual connections
3. **Feature Detection**: Analyze MLP activations to identify what features are being computed
4. **Circuit Discovery**: Compare activations across different prompts to identify consistent patterns

### Data Processing Notes

- Each activation tensor represents one forward pass step
- Multiple activations per hook indicate multi-token generation
- Shape information is crucial for proper tensor manipulation
- Hook IDs provide traceability back to specific model components
