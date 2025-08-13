# NeuroScope Experiments Compendium

This document compiles all suggested NeuroScope experiments from across the project documentation, organized by category and complexity level.

## Core Mechanistic Interpretability Experiments

### 1. Causal Tracing for Known Phenomena
**Source**: experiments.md  
**Goal**: Verify whether canonical small-model circuits (like IOI or indirect object tracking) have analogues in gpt-oss-20b.

**Implementation**:
- Patch activations from "correct" runs into "incorrect" runs and measure performance recovery
- Layer/head attribution via residual stream interventions
- Use NeuroScope's circuit finder to identify candidate head roles based on attention patterns

**NeuroScope Integration**:
```smalltalk
"Identify and test IOI circuits in larger models"
model := TransformerModel fromHuggingFace: 'gpt-oss-20b'.
circuitFinder := CircuitFinder for: model.

ioiCircuit := circuitFinder 
    findCircuit: 'indirect_object_identification'
    examples: Dataset loadIOIExamples.

"Test causal relationships"
patcher := ActivationPatcher for: model.
results := patcher testCausalRelationship: ioiCircuit.
```

### 2. Feature Localization & Ablation
**Source**: experiments.md  
**Goal**: Map specific neurons or subspaces to high-level features.

**Implementation**:
- Use activation vectors for known features (country names, code syntax) and run PCA or probing classifiers
- Zero or randomize neuron activations to measure degradation
- Apply NeuroScope's neuron analyzer to interpret high-dimensional projections

**NeuroScope Integration**:
```smalltalk
"Localize and ablate specific features"
analyzer := NeuronAnalyzer for: model.
countryNeurons := analyzer findNeuronsFor: #countryNames.

"Test ablation effects"
ablator := NeuronAblator for: model.
results := ablator testAblation: countryNeurons on: Dataset loadGeographyTest.
```

### 3. Multi-Token Steering
**Source**: experiments.md  
**Goal**: See if steering vectors (activation additions) work better when applied across multiple tokens.

**Implementation**:
- Compare one-token vs. distributed-token steering for style, bias, or factual recall
- Use NeuroScope to suggest where in the sequence features are most semantically dense

**NeuroScope Integration**:
```smalltalk
"Multi-token steering experiments"
steeringHook := ActivationHook
    layer: 10
    component: #residual
    action: [:activations :tokenPosition |
        tokenPosition > 5 ifTrue: [
            "Apply stronger steering to later tokens"
            steeringVector := SteeringVectors load: #formalStyle.
            activations := activations + (steeringVector * 1.5).
        ].
        activations].

model hookManager addHook: steeringHook.
```

### 4. Circuit Growth with Scale
**Source**: experiments.md  
**Goal**: Compare circuit complexity to smaller checkpoints of the same architecture.

**Implementation**:
- Same prompt set, same analysis pipeline, measure number of heads/MLPs required for task success
- Use NeuroScope to analyze patterns of reuse vs. specialization across scale

**NeuroScope Integration**:
```smalltalk
"Compare circuits across model scales"
smallModel := TransformerModel fromHuggingFace: 'gpt2-medium'.
largeModel := TransformerModel fromHuggingFace: 'gpt-oss-20b'.

comparator := CircuitComparator 
    smallModel: smallModel
    largeModel: largeModel.

scaleAnalysis := comparator analyzeCircuitGrowth: #arithmeticReasoning.
```

### 5. Cross-Domain Feature Entanglement
**Source**: experiments.md  
**Goal**: Detect whether a neuron/feature responds to semantically unrelated inputs.

**Implementation**:
- Activation similarity search across domains (math, code, prose)
- Use NeuroScope to hypothesize shared latent features and potential multi-task roles

**NeuroScope Integration**:
```smalltalk
"Cross-domain entanglement analysis"
entanglementAnalyzer := FeatureEntanglementAnalyzer for: model.
entangledFeatures := entanglementAnalyzer 
    findEntanglement: #(#mathematics #programming #literature).

"Visualize entanglement patterns"
visualizer := EntanglementVisualizer for: entangledFeatures.
visualizer openBrowser.
```

## Advanced Circuit Analysis Experiments

### 6. Circuit-Based Weight Editing
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Directly modify weights that implement specific computational circuits.

**Implementation**:
```smalltalk
"Identify and modify a specific circuit"
model := TransformerModel fromHuggingFace: 'gpt-oss-20b'.
circuitFinder := CircuitFinder for: model.

"Discover circuit responsible for gender bias in pronoun resolution"
biasCircuit := circuitFinder 
    findCircuit: 'gender_pronoun_bias'
    examples: Dataset loadGenderBiasExamples.

"Extract and modify weights"
circuitWeights := biasCircuit extractWeights.
debiasTransform := WeightTransform 
    removeGenderBias: circuitWeights
    preservingCapability: #pronounResolution.

model applyWeightTransform: debiasTransform to: biasCircuit.
```

### 7. Activation Steering and Control
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Use insights about activation patterns to steer model behavior without changing weights.

**Implementation**:
```smalltalk
"Create persistent activation steering"
analyzer := NeuronAnalyzer for: model.
formalityNeurons := analyzer findNeuronsFor: #textFormality.

formalityHook := ActivationHook
    layer: formalityNeurons layer
    component: #mlp
    condition: [:context | context taskType = #formalWriting]
    action: [:activations |
        formalityNeurons do: [:neuronId |
            current := activations at: neuronId.
            activations at: neuronId put: (current * 1.5).
        ].
        activations].

model hookManager addPersistentHook: formalityHook.
```

### 8. Knowledge Injection and Editing
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Modify specific factual knowledge while preserving general capabilities.

**Implementation**:
```smalltalk
"Update factual knowledge based on circuit analysis"
knowledgeEditor := KnowledgeEditor for: model.

factCircuit := knowledgeEditor 
    findFactualKnowledgeCircuit: 'The capital of France is Paris'
    layer: 12.

knowledgeEditor 
    updateFact: 'The capital of France is Lyon'
    circuit: factCircuit
    preservingRelated: #(#geography #europeanCities).
```

### 9. Capability Transfer and Enhancement
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Transfer discovered circuits between models or enhance existing capabilities.

**Implementation**:
```smalltalk
"Transfer mathematical reasoning circuit"
sourceModel := TransformerModel fromHuggingFace: 'gpt-oss-20b'.
targetModel := TransformerModel fromHuggingFace: 'gpt2-medium'.

circuitTransfer := CircuitTransfer 
    from: sourceModel
    to: targetModel.

mathCircuit := CircuitFinder for: sourceModel.
mathCircuit findCircuit: #arithmeticReasoning.

adaptedCircuit := circuitTransfer 
    adaptCircuit: mathCircuit
    targetArchitecture: targetModel architecture.

targetModel installCircuit: adaptedCircuit at: #layer10.
```

## Composable Circuit Ecosystem Experiments

### 10. Circuit Library Development
**Source**: musings/02-composable-circuit-ecosystem.md  
**Goal**: Build comprehensive libraries of discovered and validated circuits.

**Implementation**:
```smalltalk
"Build circuit library with comprehensive metadata"
CircuitLibrary class>>initialize
    SyntaxCircuits := Dictionary new.
    SyntaxCircuits 
        at: #subjectVerbAgreement put: (self loadCircuit: 'syntax/subject_verb_agreement.circuit');
        at: #pronounResolution put: (self loadCircuit: 'syntax/pronoun_resolution.circuit').
    
    SemanticCircuits := Dictionary new.
    SemanticCircuits
        at: #entityRecognition put: (self loadCircuit: 'semantics/entity_recognition.circuit');
        at: #relationExtraction put: (self loadCircuit: 'semantics/relation_extraction.circuit').
```

### 11. Automated Circuit Discovery
**Source**: musings/02-composable-circuit-ecosystem.md  
**Goal**: Develop automated tools for discovering new circuits in existing models.

**Implementation**:
```smalltalk
"Automated circuit discovery pipeline"
CircuitDiscovery class>>discoverCircuitsIn: model forCapability: capability
    discovery := self new.
    discovery
        model: model;
        targetCapability: capability;
        searchStrategy: #exhaustiveWithPruning.
    
    hypotheses := discovery generateHypotheses.
    
    validatedCircuits := OrderedCollection new.
    hypotheses do: [:hypothesis |
        testResults := discovery testHypothesis: hypothesis.
        testResults isValid ifTrue: [
            circuit := discovery extractCircuit: hypothesis.
            validatedCircuits add: circuit.
        ].
    ].
    
    ^validatedCircuits.
```

### 12. Circuit Composition Framework
**Source**: musings/02-composable-circuit-ecosystem.md  
**Goal**: Create sophisticated systems for combining circuits into functional AI systems.

**Implementation**:
```smalltalk
"Compose circuits into specialized model"
ModelComposer class>>buildSentimentAnalyzer
    composer := self new.
    
    composer
        addCircuit: (CircuitLibrary load: #tokenization) priority: 1;
        addCircuit: (CircuitLibrary load: #emotionDetection) priority: 3;
        addCircuit: (CircuitLibrary load: #sentimentPolarity) priority: 3.
    
    composer
        resolveDependencies;
        optimizeDataFlow;
        validateCompatibility.
    
    ^composer compile.
```

### 13. Quality Assurance and Validation
**Source**: musings/02-composable-circuit-ecosystem.md  
**Goal**: Implement comprehensive testing to ensure circuit reliability.

**Implementation**:
```smalltalk
"Multi-level circuit validation"
CircuitQualityAssurance class>>validateCircuit: circuit
    qa := self new circuit: circuit.
    
    functionalResults := qa testFunctionality.
    performanceResults := qa benchmarkPerformance.
    interpretabilityResults := qa validateInterpretability.
    safetyResults := qa testSafety.
    
    report := QualityReport new
        functional: functionalResults;
        performance: performanceResults;
        interpretability: interpretabilityResults;
        safety: safetyResults;
        overallScore: (qa calculateOverallScore).
    
    ^report.
```

## MLX Engine Integration Experiments

### 14. REST API Activation Capture
**Source**: README_NEUROSCOPE_INTEGRATION.md, demo/README.md  
**Goal**: Capture neural activations during text generation via REST API.

**Implementation**:
```python
# REST API activation capture
activation_hooks = [
    {
        'layer_name': 'transformer.h.5',
        'component': 'residual',
        'hook_id': 'layer_5_residual'
    }
]

result = client.generate_with_activations(
    messages=[{"role": "user", "content": "Explain transformers"}],
    activation_hooks=activation_hooks,
    max_tokens=100
)

activations = result['activations']
```

**NeuroScope Integration**:
```smalltalk
"Use REST API for activation capture"
client := LMStudioRESTClient default.
client loadModel: '/path/to/model'.

hooks := client createCircuitAnalysisHooks: 24.
result := client generateWithActivations: messages hooks: hooks.

activations := client convertActivationsForNeuroScope: (result at: #activations).
```

### 15. Circuit Analysis via MLX Engine
**Source**: demo/README.md, IMPLEMENTATION_SUMMARY.md  
**Goal**: Perform comprehensive circuit analysis using MLX Engine's activation hooks.

**Implementation**:
- Mathematical Reasoning: Arithmetic computation circuits
- Factual Recall: Knowledge retrieval mechanisms  
- Creative Writing: Generative and creative circuits
- Attention Patterns: Multi-head attention analysis
- Residual Stream: Information flow tracking

### 16. Streaming Activation Analysis
**Source**: README_NEUROSCOPE_INTEGRATION.md  
**Goal**: Real-time activation capture during streaming generation.

**Implementation**:
```python
# Enable streaming for real-time analysis
result = client.generate_with_activations(
    messages=[{"role": "user", "content": "Write a story"}],
    activation_hooks=hooks,
    stream=True,
    max_tokens=200
)
```

## Safety and Alignment Experiments

### 17. Safety-Oriented Modifications
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Apply interpretability insights to improve model safety and alignment.

**Implementation**:
```smalltalk
"Remove harmful capabilities while preserving beneficial ones"
safetyEditor := SafetyEditor for: model.

harmfulCircuits := safetyEditor 
    findHarmfulCircuits: Dataset loadHarmfulPrompts
    categories: #(#violence #misinformation #toxicity).

harmfulCircuits do: [:circuit |
    safetyHook := InterventionHook
        layer: circuit layer
        condition: [:context | safetyEditor detectsHarmfulIntent: context]
        action: [:activations | safetyEditor redirectToSafeOutput: activations].
    
    model hookManager addHook: safetyHook.
].
```

### 18. Interpretability-Guided Fine-tuning
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Use circuit insights to guide more effective fine-tuning.

**Implementation**:
```smalltalk
"Fine-tune only circuits relevant to target task"
sentimentCircuits := CircuitFinder for: model.
sentimentCircuits findCircuitsFor: #sentimentAnalysis.

trainer := CircuitAwareTrainer 
    model: model
    targetCircuits: sentimentCircuits.

trainer addObjective: #taskPerformance weight: 0.7.
trainer addObjective: #circuitCoherence weight: 0.2.
trainer addObjective: #generalCapabilityPreservation weight: 0.1.

trainedModel := trainer train: taskDataset.
```

## Experimental Validation Framework

### 19. Pre-Modification Analysis
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Analyze potential impacts before making modifications.

**Implementation**:
```smalltalk
"Assess risks of proposed modification"
safetyAnalyzer := ModificationSafetyAnalyzer for: model.

riskAssessment := safetyAnalyzer 
    assessModification: proposedCircuitEdit
    categories: #(#capabilityLoss #behaviorChange #safetyRisk).

riskAssessment overallRisk < #medium ifTrue: [
    model applyModification: proposedCircuitEdit.
    monitor := ModificationMonitor for: model.
    monitor trackChanges: #(#performance #safety #capabilities).
].
```

### 20. Post-Modification Validation
**Source**: musings/01-model-modification-pathways.md  
**Goal**: Comprehensive validation after modification.

**Implementation**:
```smalltalk
"Comprehensive validation after modification"
validator := ComprehensiveValidator for: model.

validationResults := Dictionary new.
validationResults at: #originalPerformance put:
    (validator testPerformance: originalTaskDataset).
validationResults at: #safety put:
    (validator testSafety: safetyTestSuite).
validationResults at: #generalCapabilities put:
    (validator testGeneralCapabilities: generalTestSuite).

ValidationReporter generateReport: validationResults.
```

## Implementation Priority Levels

### High Priority (Immediate Implementation)
1. **Causal Tracing for Known Phenomena** - Foundational for validating NeuroScope
2. **Feature Localization & Ablation** - Core interpretability functionality
3. **REST API Activation Capture** - Essential for MLX Engine integration
4. **Circuit Analysis via MLX Engine** - Validates the complete workflow

### Medium Priority (Next Phase)
5. **Multi-Token Steering** - Advanced steering capabilities
6. **Circuit Growth with Scale** - Important for understanding scaling
7. **Automated Circuit Discovery** - Accelerates research
8. **Safety-Oriented Modifications** - Critical for responsible AI

### Long-term Goals (Future Development)
9. **Circuit Library Development** - Ecosystem building
10. **Circuit Composition Framework** - Advanced composability
11. **Knowledge Injection and Editing** - Precise model modification
12. **Capability Transfer and Enhancement** - Cross-model capabilities

## Getting Started

### Prerequisites
- NeuroScope environment with Smalltalk
- MLX Engine with activation hooks (see IMPLEMENTATION_SUMMARY.md)
- gpt-oss-20b model or similar large transformer
- Sufficient computational resources (32GB+ RAM recommended)

### First Experiments
1. Start with **REST API Activation Capture** to validate the integration
2. Implement **Feature Localization & Ablation** for basic interpretability
3. Try **Causal Tracing** on simple, well-understood phenomena
4. Gradually work toward more complex circuit analysis

### Resources
- `examples/neuroscope_integration_example.py` - Comprehensive usage examples
- `demo/README.md` - Complete demo documentation
- `README_NEUROSCOPE_INTEGRATION.md` - Integration guide
- `IMPLEMENTATION_SUMMARY.md` - Technical implementation details

This compendium provides a roadmap for mechanistic interpretability research using NeuroScope, from basic activation analysis to advanced circuit modification and composition.