# NeuroScope MLX Engine Demo

This directory contains the demonstration scripts and generated data for the NeuroScope integration with MLX Engine.

## Files

### Demo Scripts
- **`run_neuroscope_demo.py`** - Main demo runner that executes all tests
- **`demo_neuroscope_rest_interface.py`** - Core demo implementation showing REST API usage
- **`run_neuroscope_demo.log`** - Latest demo execution log

### Data Directory
- **`data/`** - Contains generated activation data and format documentation

## Running the Demo

From the project root directory:

```bash
python demo/run_neuroscope_demo.py
```

This will:
1. Run basic functionality tests
2. Demonstrate activation capture
3. Show circuit analysis capabilities
4. Generate data files in `demo/data/`
5. Create comprehensive format documentation

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

## Integration with NeuroScope

The generated data files provide concrete examples of:
- Real activation tensor formats
- Multi-layer circuit analysis results
- Temporal dynamics during text generation
- Component isolation (attention vs MLP)

These files serve as reference implementations for NeuroScope integration.