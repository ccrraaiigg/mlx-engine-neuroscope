# Sophisticated Circuit Discovery Implementation Design

## Overview

This document outlines the design for implementing advanced mechanistic interpretability techniques in the MLX Engine Neuroscope system. The goal is to enhance circuit discovery capabilities beyond basic activation capture to include sophisticated causal intervention analysis, feature localization, and comprehensive visualization.

## Current State Analysis

### Existing Implementation
- **Basic Circuit Discovery**: Currently implemented in `mcp-server/src/tools/discover_circuits.js`
- **Activation Capture**: Working activation hooks system in `mlx_engine/activation_hooks.py`
- **Visualization**: 3D force-directed graph visualization in `mcp-server/src/tools/circuit_diagram.js`
- **MLX Engine Integration**: Functional API at `http://localhost:50111/v1/chat/completions/with_activations`

### Current Limitations
- No activation patching for causal intervention
- Basic confidence scoring based only on activation count
- Limited feature localization capabilities
- No systematic ablation testing framework
- Minimal attention pattern analysis

## Implementation Plan

### 1. Activation Patching (High Priority)

**Objective**: Implement causal intervention analysis to identify critical circuit components.

**Technical Approach**:
- **Patch Generation**: Create clean and corrupted activation pairs
- **Intervention Points**: Target specific layers, attention heads, and MLP components
- **Causal Metrics**: Measure logit differences and probability shifts
- **Statistical Validation**: Use multiple runs with different corruption methods

**Implementation Details**:
```javascript
// New tool: activation_patching.js
const patchingMethods = {
  'zero_ablation': (activations) => activations.map(() => 0),
  'mean_ablation': (activations) => activations.map(() => mean(activations)),
  'noise_injection': (activations, noise_level) => addGaussianNoise(activations, noise_level),
  'random_replacement': (activations, dataset) => sampleFromDataset(dataset)
};
```

**MLX Engine Extensions**:
- Add `/v1/interventions/patch_activations` endpoint
- Implement activation replacement during forward pass
- Support batch processing for statistical significance

### 2. Enhanced Causal Tracing (High Priority)

**Objective**: Develop sophisticated algorithms with noise injection and gradient-based attribution.

**Technical Components**:
- **Gradient Attribution**: Compute activation gradients with respect to output logits
- **Integrated Gradients**: Path integration for more stable attributions
- **Noise Baselines**: Multiple corruption strategies for robust causal estimates
- **Layer-wise Analysis**: Systematic scanning across all transformer layers

**Algorithm Design**:
```python
def enhanced_causal_trace(model, clean_prompt, corrupted_prompt, target_token):
    # 1. Compute clean and corrupted activations
    clean_acts = capture_activations(model, clean_prompt)
    corrupt_acts = capture_activations(model, corrupted_prompt)
    
    # 2. For each intervention point
    causal_effects = {}
    for layer in range(model.num_layers):
        for component in ['attention', 'mlp']:
            # Patch corrupted with clean activations
            patched_logits = patch_and_forward(corrupt_acts, clean_acts, layer, component)
            causal_effects[f"{layer}_{component}"] = compute_causal_effect(patched_logits, target_token)
    
    return causal_effects
```

### 3. Advanced Feature Localization (High Priority)

**Objective**: Create feature localization using sparse autoencoders and dictionary learning.

**Technical Approach**:
- **Sparse Autoencoders**: Train on activation datasets to discover interpretable features
- **Dictionary Learning**: Use K-SVD or similar algorithms for feature extraction
- **Probing Classifiers**: Train linear probes to identify specific concepts
- **Feature Visualization**: Generate examples that maximally activate discovered features

**Implementation Structure**:
```javascript
// New tool: feature_localization.js
const localizationMethods = {
  'sparse_autoencoder': {
    architecture: 'encoder-decoder with L1 sparsity',
    training_data: 'activation_datasets',
    output: 'interpretable_feature_directions'
  },
  'dictionary_learning': {
    algorithm: 'K-SVD or Online Dictionary Learning',
    sparsity_constraint: 'L0 or L1 regularization',
    output: 'sparse_feature_dictionary'
  },
  'probing_classifiers': {
    architecture: 'linear_probe or shallow_mlp',
    training_labels: 'concept_annotations',
    output: 'concept_localization_maps'
  }
};
```

### 4. Comprehensive Attention Analysis (Medium Priority)

**Objective**: Build head-specific circuit discovery with attention pattern analysis.

**Features**:
- **Head-level Analysis**: Individual attention head behavior tracking
- **Pattern Classification**: Identify induction heads, copying heads, etc.
- **Cross-layer Dependencies**: Track attention flow across layers
- **Token-level Attribution**: Fine-grained attention weight analysis

**Visualization Enhancements**:
- Attention flow diagrams
- Head-specific heatmaps
- Token-to-token attention paths
- Layer-wise attention summaries

### 5. Interactive Circuit Visualization (Medium Priority)

**Objective**: Enhance 3D visualization with component highlighting and interaction.

**New Features**:
- **Component Filtering**: Show/hide specific layer types or components
- **Causal Strength Encoding**: Node size/color based on causal importance
- **Interactive Patching**: Click-to-patch interface for real-time intervention
- **Multi-view Support**: Side-by-side comparison of different circuits

**Technical Implementation**:
```javascript
// Enhanced circuit_diagram.js
const visualizationFeatures = {
  'interactive_patching': {
    'click_handler': 'patch_component_on_click',
    'real_time_updates': 'update_downstream_effects',
    'undo_redo': 'intervention_history_stack'
  },
  'causal_encoding': {
    'node_size': 'proportional_to_causal_effect',
    'edge_thickness': 'information_flow_strength',
    'color_scheme': 'causal_importance_gradient'
  }
};
```

### 6. Residual Stream Tracking (Medium Priority)

**Objective**: Track information flow across layers and components.

**Technical Components**:
- **Stream Decomposition**: Separate attention and MLP contributions
- **Information Flow Metrics**: Quantify information transfer between layers
- **Bottleneck Analysis**: Identify information processing bottlenecks
- **Temporal Dynamics**: Track how information evolves during generation

### 7. Systematic Ablation Framework (Low Priority)

**Objective**: Create comprehensive testing framework for circuit validation.

**Framework Components**:
- **Automated Test Generation**: Create test cases for discovered circuits
- **Performance Metrics**: Accuracy, F1, causal necessity/sufficiency
- **Statistical Testing**: Significance tests for circuit importance
- **Robustness Analysis**: Test circuit behavior across different inputs

## File Structure

```
mcp-server/src/tools/
├── activation_patching.js          # New: Causal intervention analysis
├── enhanced_causal_tracing.js      # New: Advanced causal algorithms
├── feature_localization.js         # Enhanced: Sparse autoencoders + dictionary learning
├── attention_analysis.js           # New: Head-specific analysis
├── circuit_diagram.js              # Enhanced: Interactive visualization
├── residual_stream_tracking.js     # New: Information flow analysis
└── ablation_framework.js           # New: Systematic testing

mlx_engine/
├── activation_hooks.py             # Enhanced: Support for patching
├── causal_interventions.py         # New: Intervention implementations
├── feature_extractors.py           # New: Autoencoder and dictionary learning
└── api_server.py                   # Enhanced: New endpoints
```

## API Extensions

### New MLX Engine Endpoints

```python
# Activation Patching
POST /v1/interventions/patch_activations
{
  "prompt": "What is the capital of France?",
  "intervention_points": [
    {"layer": 10, "component": "attention", "method": "zero_ablation"},
    {"layer": 15, "component": "mlp", "method": "mean_ablation"}
  ],
  "baseline_prompt": "What is the capital of Spain?"
}

# Feature Localization
POST /v1/analysis/localize_features
{
  "feature_name": "country_capitals",
  "layer_range": {"start": 8, "end": 16},
  "method": "sparse_autoencoder",
  "training_prompts": ["What is the capital of...", ...]
}

# Enhanced Causal Tracing
POST /v1/analysis/causal_trace
{
  "clean_prompt": "The capital of France is Paris",
  "corrupted_prompt": "The capital of France is London",
  "target_token": "Paris",
  "methods": ["gradient_attribution", "integrated_gradients"]
}
```

## Success Metrics

### Quantitative Metrics
- **Circuit Precision**: Percentage of discovered components that are causally necessary
- **Circuit Recall**: Percentage of ground-truth circuit components discovered
- **Intervention Accuracy**: Success rate of targeted interventions
- **Feature Interpretability**: Human evaluation scores for discovered features

### Qualitative Metrics
- **Visualization Clarity**: User feedback on circuit diagram comprehensibility
- **Analysis Depth**: Complexity and insight quality of discovered circuits
- **Tool Usability**: Ease of use for mechanistic interpretability researchers

## Implementation Timeline

### Phase 1 (Weeks 1-2): Core Infrastructure
- Implement activation patching framework
- Enhance MLX Engine with intervention endpoints
- Create basic causal tracing algorithms

### Phase 2 (Weeks 3-4): Advanced Analysis
- Develop feature localization tools
- Implement attention pattern analysis
- Enhance visualization with interactive features

### Phase 3 (Weeks 5-6): Integration and Testing
- Add residual stream tracking
- Create ablation testing framework
- Comprehensive testing and validation

## Dependencies

### Python Packages
- `scikit-learn`: For dictionary learning and PCA
- `torch`: For autoencoder training (if not using MLX)
- `numpy`: For numerical computations
- `scipy`: For statistical testing

### JavaScript Packages
- `d3.js`: Enhanced for interactive visualizations
- `three.js`: For 3D circuit diagrams
- `plotly.js`: For statistical plots and heatmaps

## Risk Mitigation

### Technical Risks
- **Performance**: Large activation tensors may cause memory issues
  - *Mitigation*: Implement streaming and chunked processing
- **Accuracy**: Causal estimates may be noisy
  - *Mitigation*: Use multiple baselines and statistical validation

### Integration Risks
- **API Compatibility**: New endpoints may break existing tools
  - *Mitigation*: Maintain backward compatibility and versioning
- **Visualization Complexity**: Too many features may overwhelm users
  - *Mitigation*: Progressive disclosure and customizable interfaces

## Future Extensions

- **Multi-model Support**: Extend beyond GPT-OSS-20B to other architectures
- **Real-time Analysis**: Live circuit discovery during model inference
- **Collaborative Features**: Multi-user circuit exploration and annotation
- **Safety Applications**: Circuit-based model editing and alignment

---

*This design document serves as a comprehensive roadmap for implementing sophisticated circuit discovery capabilities. Each component is designed to be modular and extensible, allowing for incremental development and testing.*