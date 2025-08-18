# NeuroScope REST Interface Demo

This demo shows how NeuroScope will interact with the MLX Engine REST API for mechanistic interpretability analysis. It provides a comprehensive, all-in-one demonstration of the complete workflow.

## Overview

The demo is a single integrated script that demonstrates the complete workflow for NeuroScope integration:

1. **`demo_neuroscope_rest_interface.py`** - Complete integrated demo with all functionality
2. **`../code/neuroscope_api_reference.py`** - Comprehensive API reference and examples  
3. **`data/`** - Generated activation data and format documentation

## What's Demonstrated

### 1. Complete Integrated Workflow
- **Basic Engine Test**: Core model functionality validation
- **Basic REST Interface**: Health checks, model loading, standard generation
- **Activation Capture**: Generate text while capturing neural activations
- **Circuit Analysis**: Comprehensive analysis across multiple layers
- **Streaming Concept**: Real-time activation capture during generation
- **Integration Workflow**: Complete NeuroScope integration demonstration

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

From the demo directory:

```bash
python demo_neuroscope_rest_interface.py
```

This single script runs the complete integrated demo:
1. Basic Engine Test (core model functionality)
2. Basic REST Interface (API validation)
3. Activation Capture (neural activation analysis)
4. Circuit Analysis (multi-layer analysis)
5. Streaming Concept (real-time processing)
6. Integration Workflow (complete NeuroScope workflow)

### Additional Components

```bash
# API reference and examples (optional)
python ../code/neuroscope_api_reference.py
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

## Demo Architecture

### Integrated Testing Approach

**NeuroScope Demo - Complete Integration Testing**
- **Purpose**: Validates the complete NeuroScope integration workflow in a single script
- **Includes**:
  - Core MLX engine functionality validation (Basic Engine Test)
  - REST API validation (Basic REST Interface)
  - Neural activation capture (Activation Capture)
  - Multi-layer circuit analysis (Circuit Analysis)
  - Real-time processing (Streaming Concept)
  - Complete workflow demonstration (Integration Workflow)
- **Benefits**: 
  - Single model load for all tests (efficient)
  - Comprehensive end-to-end validation
  - Real-time console output with complete logging
  - All functionality in one place

**Key Insight**: This unified approach eliminates redundancy and provides a complete testing experience in a single script execution.

## Troubleshooting

### Common Issues
1. **Model Not Found**: Ensure model is downloaded and path is correct
2. **Memory Issues**: Reduce model size or increase system RAM
3. **Port Conflicts**: Change API server port if 50111 is in use
4. **Dependencies**: Install all required packages with pip

### Testing Strategy
- **Use**: `python demo_neuroscope_rest_interface.py` for comprehensive testing
- **Benefit**: All functionality validated in a single efficient run with detailed logging
- **Debugging**: Detailed debug output shows exactly where any issues occur

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