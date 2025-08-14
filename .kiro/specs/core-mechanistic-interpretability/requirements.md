# Requirements Document

## Introduction

This specification covers the implementation of core mechanistic interpretability experiments for the MLX Engine with NeuroScope integration. These experiments form the foundation for understanding transformer model internals through causal tracing, feature localization, multi-token steering, circuit growth analysis, and cross-domain feature entanglement detection.

**Note:** All capabilities defined in this specification are accessible through the **Mechanistic Interpretability MCP Server**, enabling LLM agents to perform these complex interpretability experiments through standardized MCP tools without requiring direct implementation knowledge.

## Requirements

### Requirement 1: Causal Tracing for Known Phenomena

**User Story:** As a mechanistic interpretability researcher, I want to verify whether canonical small-model circuits (like IOI or indirect object tracking) have analogues in larger models like gpt-oss-20b, so that I can validate circuit universality across model scales.

#### Acceptance Criteria

1. WHEN I provide a known circuit pattern from small models THEN the system SHALL identify candidate analogous circuits in larger models
2. WHEN I patch activations from "correct" runs into "incorrect" runs THEN the system SHALL measure performance recovery and quantify causal relationships
3. WHEN I perform layer/head attribution via residual stream interventions THEN the system SHALL provide attribution scores for each component
4. WHEN I use NeuroScope's circuit finder THEN the system SHALL identify candidate head roles based on attention patterns
5. IF the circuit exists in the larger model THEN the system SHALL provide confidence scores and validation metrics

### Requirement 2: Feature Localization and Ablation

**User Story:** As a researcher studying model internals, I want to map specific neurons or subspaces to high-level features (like country names or code syntax), so that I can understand how models represent and process different types of information.

#### Acceptance Criteria

1. WHEN I provide examples of a specific feature type THEN the system SHALL identify neurons or subspaces that activate for those features
2. WHEN I apply PCA or probing classifiers to activation vectors THEN the system SHALL provide interpretable feature representations
3. WHEN I zero or randomize neuron activations THEN the system SHALL measure performance degradation and identify critical neurons
4. WHEN I use NeuroScope's neuron analyzer THEN the system SHALL interpret high-dimensional projections and provide semantic labels
5. WHEN I ablate specific neurons THEN the system SHALL preserve model functionality for unrelated tasks

### Requirement 3: Multi-Token Steering Implementation

**User Story:** As a model behavior researcher, I want to test whether steering vectors work better when applied across multiple tokens rather than single tokens, so that I can develop more effective model control techniques.

#### Acceptance Criteria

1. WHEN I apply steering vectors to single tokens THEN the system SHALL measure behavioral changes and effectiveness
2. WHEN I apply steering vectors across multiple tokens THEN the system SHALL compare effectiveness against single-token steering
3. WHEN I test different steering strategies THEN the system SHALL support style, bias, and factual recall modifications
4. WHEN I use NeuroScope to analyze semantic density THEN the system SHALL suggest optimal token positions for steering
5. WHEN I apply distributed steering THEN the system SHALL maintain coherent text generation quality

### Requirement 4: Circuit Growth Analysis with Scale

**User Story:** As a scaling researcher, I want to compare circuit complexity between different model sizes using the same architecture, so that I can understand how computational circuits evolve with model scale.

#### Acceptance Criteria

1. WHEN I analyze the same task across different model sizes THEN the system SHALL use identical prompt sets and analysis pipelines
2. WHEN I measure circuit complexity THEN the system SHALL count the number of heads/MLPs required for task success
3. WHEN I compare circuits across scales THEN the system SHALL identify patterns of reuse vs. specialization
4. WHEN I use NeuroScope for scale analysis THEN the system SHALL provide visualizations of circuit growth patterns
5. WHEN circuits grow with scale THEN the system SHALL quantify the relationship between model size and circuit complexity

### Requirement 5: Cross-Domain Feature Entanglement Detection

**User Story:** As a feature analysis researcher, I want to detect whether neurons respond to semantically unrelated inputs across different domains, so that I can understand feature sharing and multi-task learning in transformers.

#### Acceptance Criteria

1. WHEN I analyze neuron activations across domains THEN the system SHALL perform activation similarity search across math, code, and prose domains
2. WHEN I detect feature entanglement THEN the system SHALL identify neurons that respond to multiple unrelated semantic categories
3. WHEN I use NeuroScope for entanglement analysis THEN the system SHALL hypothesize shared latent features and multi-task roles
4. WHEN I visualize entanglement patterns THEN the system SHALL provide interpretable representations of cross-domain relationships
5. WHEN entangled features are found THEN the system SHALL validate the semantic relationships through targeted experiments