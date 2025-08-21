"""Comprehensive Attention Pattern Analysis for Mechanistic Interpretability

This module implements sophisticated attention analysis techniques including:
- Head-level attention pattern classification
- Cross-layer attention dependencies
- Token-level attention attribution
- Attention circuit discovery
- Pattern visualization and interpretation
"""

from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import logging
from .activation_hooks import ActivationHookManager, ComponentType, ActivationHookSpec, CapturedActivation

logger = logging.getLogger(__name__)

class AttentionPatternType(Enum):
    """Types of attention patterns that can be identified."""
    INDUCTION = "induction"  # Copying from previous occurrences
    PREVIOUS_TOKEN = "previous_token"  # Attending to previous token
    FIRST_TOKEN = "first_token"  # Attending to first token (BOS)
    PUNCTUATION = "punctuation"  # Attending to punctuation
    SYNTACTIC = "syntactic"  # Syntactic relationships
    SEMANTIC = "semantic"  # Semantic relationships
    POSITIONAL = "positional"  # Position-based patterns
    BROADCAST = "broadcast"  # Broadcasting information
    INHIBITION = "inhibition"  # Suppressing information
    UNKNOWN = "unknown"  # Unclassified patterns

class AttentionScope(Enum):
    """Scope of attention analysis."""
    HEAD_LEVEL = "head_level"  # Individual attention heads
    LAYER_LEVEL = "layer_level"  # Entire attention layers
    CROSS_LAYER = "cross_layer"  # Dependencies across layers
    GLOBAL = "global"  # Global attention patterns

@dataclass
class AttentionHead:
    """Represents an individual attention head."""
    layer: int
    head: int
    pattern_type: AttentionPatternType
    confidence: float
    attention_weights: mx.array
    key_tokens: List[int]  # Tokens this head focuses on
    value_tokens: List[int]  # Tokens this head retrieves from
    circuit_role: str  # Role in discovered circuits
    metadata: Dict[str, Any]

@dataclass
class AttentionPattern:
    """Represents a discovered attention pattern."""
    pattern_type: AttentionPatternType
    heads: List[AttentionHead]
    strength: float
    consistency: float  # How consistent across examples
    token_positions: List[Tuple[int, int]]  # (source, target) positions
    description: str
    examples: List[str]  # Example prompts where pattern occurs

@dataclass
class CrossLayerDependency:
    """Represents dependencies between attention layers."""
    source_layer: int
    target_layer: int
    dependency_type: str  # e.g., "information_flow", "gating", "refinement"
    strength: float
    affected_heads: List[Tuple[int, int]]  # (layer, head) pairs
    mechanism: str  # Description of the dependency mechanism

@dataclass
class AttentionAnalysisResult:
    """Results of comprehensive attention analysis."""
    attention_heads: List[AttentionHead]
    attention_patterns: List[AttentionPattern]
    cross_layer_dependencies: List[CrossLayerDependency]
    token_attributions: Dict[int, Dict[str, float]]  # token_idx -> {"importance": score, "role": description}
    circuit_components: List[Dict[str, Any]]  # Attention components in discovered circuits
    pattern_statistics: Dict[str, Any]
    visualization_data: Dict[str, Any]
    execution_time_ms: int
    metadata: Dict[str, Any]

class AttentionPatternClassifier:
    """Classifies attention patterns based on attention weights."""
    
    def __init__(self):
        self.pattern_thresholds = {
            AttentionPatternType.INDUCTION: 0.3,
            AttentionPatternType.PREVIOUS_TOKEN: 0.5,
            AttentionPatternType.FIRST_TOKEN: 0.4,
            AttentionPatternType.PUNCTUATION: 0.3,
            AttentionPatternType.SYNTACTIC: 0.25,
            AttentionPatternType.SEMANTIC: 0.2
        }
    
    def classify_head_pattern(self, attention_weights: mx.array, tokens: List[int], 
                            layer: int, head: int) -> AttentionHead:
        """Classify the pattern of a single attention head."""
        seq_len = attention_weights.shape[-1]
        
        # Analyze attention distribution
        pattern_scores = self._compute_pattern_scores(attention_weights, tokens)
        
        # Determine primary pattern type
        primary_pattern = max(pattern_scores.items(), key=lambda x: x[1])
        pattern_type = primary_pattern[0]
        confidence = primary_pattern[1]
        
        # Identify key tokens this head attends to
        key_tokens = self._identify_key_tokens(attention_weights, tokens)
        value_tokens = self._identify_value_tokens(attention_weights, tokens)
        
        # Determine circuit role
        circuit_role = self._determine_circuit_role(pattern_type, attention_weights)
        
        return AttentionHead(
            layer=layer,
            head=head,
            pattern_type=pattern_type,
            confidence=confidence,
            attention_weights=attention_weights,
            key_tokens=key_tokens,
            value_tokens=value_tokens,
            circuit_role=circuit_role,
            metadata={
                "pattern_scores": pattern_scores,
                "attention_entropy": self._compute_attention_entropy(attention_weights),
                "max_attention": float(mx.max(attention_weights)),
                "attention_sparsity": self._compute_sparsity(attention_weights)
            }
        )
    
    def _compute_pattern_scores(self, attention_weights: mx.array, tokens: List[int]) -> Dict[AttentionPatternType, float]:
        """Compute scores for different attention patterns."""
        scores = {}
        seq_len = attention_weights.shape[-1]
        
        # Previous token pattern
        if seq_len > 1:
            prev_token_score = float(mx.mean(mx.diag(attention_weights, k=-1)))
            scores[AttentionPatternType.PREVIOUS_TOKEN] = prev_token_score
        
        # First token pattern
        if seq_len > 0:
            first_token_score = float(mx.mean(attention_weights[:, 0]))
            scores[AttentionPatternType.FIRST_TOKEN] = first_token_score
        
        # Induction pattern (simplified heuristic)
        induction_score = self._compute_induction_score(attention_weights, tokens)
        scores[AttentionPatternType.INDUCTION] = induction_score
        
        # Punctuation pattern
        punct_score = self._compute_punctuation_score(attention_weights, tokens)
        scores[AttentionPatternType.PUNCTUATION] = punct_score
        
        # Positional pattern
        pos_score = self._compute_positional_score(attention_weights)
        scores[AttentionPatternType.POSITIONAL] = pos_score
        
        # Default to unknown for low scores
        max_score = max(scores.values()) if scores else 0.0
        if max_score < 0.15:
            scores[AttentionPatternType.UNKNOWN] = 1.0 - max_score
        
        return scores
    
    def _compute_induction_score(self, attention_weights: mx.array, tokens: List[int]) -> float:
        """Compute score for induction head pattern."""
        # Simplified induction detection - look for attention to repeated patterns
        seq_len = len(tokens)
        if seq_len < 4:
            return 0.0
        
        induction_score = 0.0
        for i in range(2, seq_len):
            for j in range(i-2):
                if tokens[j] == tokens[i-1]:  # Found a repeat
                    # Check if attention focuses on position after the first occurrence
                    if j+1 < seq_len:
                        induction_score += float(attention_weights[i, j+1])
        
        return induction_score / max(seq_len - 2, 1)
    
    def _compute_punctuation_score(self, attention_weights: mx.array, tokens: List[int]) -> float:
        """Compute score for punctuation attention pattern."""
        # Common punctuation token IDs (this would need to be model-specific)
        punct_tokens = {13, 25, 26, 27, 28, 29, 30}  # Common punctuation IDs
        
        punct_positions = [i for i, token in enumerate(tokens) if token in punct_tokens]
        if not punct_positions:
            return 0.0
        
        punct_attention = mx.sum(attention_weights[:, punct_positions], axis=1)
        return float(mx.mean(punct_attention))
    
    def _compute_positional_score(self, attention_weights: mx.array) -> float:
        """Compute score for positional attention patterns."""
        seq_len = attention_weights.shape[-1]
        if seq_len < 3:
            return 0.0
        
        # Check for diagonal patterns, uniform patterns, etc.
        diagonal_score = float(mx.mean(mx.diag(attention_weights)))
        uniform_score = 1.0 - float(mx.std(attention_weights))
        
        return max(diagonal_score, uniform_score * 0.5)
    
    def _compute_attention_entropy(self, attention_weights: mx.array) -> float:
        """Compute entropy of attention distribution."""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs = attention_weights + eps
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        
        log_probs = mx.log(probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        return float(mx.mean(entropy))
    
    def _compute_sparsity(self, attention_weights: mx.array) -> float:
        """Compute sparsity of attention weights."""
        threshold = 0.1
        sparse_weights = attention_weights > threshold
        sparsity = 1.0 - float(mx.mean(sparse_weights))
        return sparsity
    
    def _identify_key_tokens(self, attention_weights: mx.array, tokens: List[int]) -> List[int]:
        """Identify tokens this head primarily attends to."""
        # Get top-k attended positions
        k = min(3, attention_weights.shape[-1])
        top_positions = mx.topk(mx.mean(attention_weights, axis=0), k=k)[1]
        return [tokens[int(pos)] for pos in top_positions if int(pos) < len(tokens)]
    
    def _identify_value_tokens(self, attention_weights: mx.array, tokens: List[int]) -> List[int]:
        """Identify tokens this head retrieves information from."""
        # Similar to key tokens but focuses on where information flows from
        return self._identify_key_tokens(attention_weights, tokens)
    
    def _determine_circuit_role(self, pattern_type: AttentionPatternType, attention_weights: mx.array) -> str:
        """Determine the role of this head in neural circuits."""
        role_mapping = {
            AttentionPatternType.INDUCTION: "information_copying",
            AttentionPatternType.PREVIOUS_TOKEN: "sequential_processing",
            AttentionPatternType.FIRST_TOKEN: "context_aggregation",
            AttentionPatternType.PUNCTUATION: "syntax_processing",
            AttentionPatternType.SYNTACTIC: "structural_analysis",
            AttentionPatternType.SEMANTIC: "meaning_extraction",
            AttentionPatternType.POSITIONAL: "position_encoding",
            AttentionPatternType.BROADCAST: "information_distribution",
            AttentionPatternType.INHIBITION: "information_suppression"
        }
        return role_mapping.get(pattern_type, "unknown_role")

class CrossLayerAnalyzer:
    """Analyzes dependencies and information flow between attention layers."""
    
    def analyze_dependencies(self, attention_data: Dict[str, Any]) -> List[CrossLayerDependency]:
        """Analyze cross-layer dependencies in attention patterns."""
        dependencies = []
        layers = sorted([int(layer.split('.')[-1]) for layer in attention_data.keys() 
                        if 'layers' in layer and 'self_attn' in layer])
        
        for i, source_layer in enumerate(layers[:-1]):
            for target_layer in layers[i+1:]:
                dependency = self._analyze_layer_pair(source_layer, target_layer, attention_data)
                if dependency:
                    dependencies.append(dependency)
        
        return dependencies
    
    def _analyze_layer_pair(self, source_layer: int, target_layer: int, 
                          attention_data: Dict[str, Any]) -> Optional[CrossLayerDependency]:
        """Analyze dependency between a pair of layers."""
        source_key = f"model.layers.{source_layer}.self_attn"
        target_key = f"model.layers.{target_layer}.self_attn"
        
        if source_key not in attention_data or target_key not in attention_data:
            return None
        
        source_attn = attention_data[source_key]
        target_attn = attention_data[target_key]
        
        # Compute dependency strength using attention pattern correlation
        dependency_strength = self._compute_dependency_strength(source_attn, target_attn)
        
        if dependency_strength > 0.3:  # Threshold for significant dependency
            dependency_type = self._classify_dependency_type(source_attn, target_attn)
            affected_heads = self._identify_affected_heads(source_attn, target_attn)
            mechanism = self._describe_mechanism(dependency_type, source_layer, target_layer)
            
            return CrossLayerDependency(
                source_layer=source_layer,
                target_layer=target_layer,
                dependency_type=dependency_type,
                strength=dependency_strength,
                affected_heads=affected_heads,
                mechanism=mechanism
            )
        
        return None
    
    def _compute_dependency_strength(self, source_attn: mx.array, target_attn: mx.array) -> float:
        """Compute strength of dependency between attention layers."""
        # Simplified correlation-based dependency measure
        source_flat = mx.reshape(source_attn, (-1,))
        target_flat = mx.reshape(target_attn, (-1,))
        
        # Compute correlation
        source_mean = mx.mean(source_flat)
        target_mean = mx.mean(target_flat)
        
        numerator = mx.sum((source_flat - source_mean) * (target_flat - target_mean))
        denominator = mx.sqrt(mx.sum((source_flat - source_mean)**2) * mx.sum((target_flat - target_mean)**2))
        
        correlation = numerator / (denominator + 1e-8)
        return float(mx.abs(correlation))
    
    def _classify_dependency_type(self, source_attn: mx.array, target_attn: mx.array) -> str:
        """Classify the type of dependency between layers."""
        # Simplified classification based on attention patterns
        source_entropy = self._compute_entropy(source_attn)
        target_entropy = self._compute_entropy(target_attn)
        
        if target_entropy < source_entropy * 0.8:
            return "information_refinement"
        elif target_entropy > source_entropy * 1.2:
            return "information_expansion"
        else:
            return "information_flow"
    
    def _compute_entropy(self, attention: mx.array) -> float:
        """Compute entropy of attention distribution."""
        eps = 1e-8
        probs = attention + eps
        probs = probs / mx.sum(probs, axis=-1, keepdims=True)
        log_probs = mx.log(probs)
        entropy = -mx.sum(probs * log_probs, axis=-1)
        return float(mx.mean(entropy))
    
    def _identify_affected_heads(self, source_attn: mx.array, target_attn: mx.array) -> List[Tuple[int, int]]:
        """Identify which heads are involved in the dependency."""
        # Simplified implementation - return first few heads
        num_heads = min(source_attn.shape[0], target_attn.shape[0], 4)
        return [(0, i) for i in range(num_heads)]
    
    def _describe_mechanism(self, dependency_type: str, source_layer: int, target_layer: int) -> str:
        """Describe the mechanism of the dependency."""
        mechanisms = {
            "information_refinement": f"Layer {target_layer} refines attention patterns from layer {source_layer}",
            "information_expansion": f"Layer {target_layer} expands on information from layer {source_layer}",
            "information_flow": f"Information flows from layer {source_layer} to layer {target_layer}"
        }
        return mechanisms.get(dependency_type, f"Unknown dependency between layers {source_layer} and {target_layer}")

class AttentionAnalyzer:
    """Main class for comprehensive attention analysis."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.pattern_classifier = AttentionPatternClassifier()
        self.cross_layer_analyzer = CrossLayerAnalyzer()
        self.hook_manager = ActivationHookManager(model)
    
    def analyze_attention_patterns(self, prompt: str, layers: Optional[List[int]] = None,
                                 scope: AttentionScope = AttentionScope.HEAD_LEVEL) -> AttentionAnalysisResult:
        """Perform comprehensive attention pattern analysis."""
        import time
        start_time = time.time()
        
        # Tokenize prompt
        from mlx_engine import tokenize
        tokens = tokenize(self.model, prompt)
        
        # Set up attention hooks
        if layers is None:
            layers = list(range(min(12, getattr(self.model, 'num_layers', 12))))  # Default to first 12 layers
        
        attention_hooks = self._create_attention_hooks(layers)
        
        # Capture attention activations
        attention_data = self._capture_attention_activations(prompt, attention_hooks)
        
        # Analyze attention heads
        attention_heads = []
        if scope in [AttentionScope.HEAD_LEVEL, AttentionScope.GLOBAL]:
            attention_heads = self._analyze_attention_heads(attention_data, tokens)
        
        # Identify attention patterns
        attention_patterns = self._identify_attention_patterns(attention_heads)
        
        # Analyze cross-layer dependencies
        cross_layer_dependencies = []
        if scope in [AttentionScope.CROSS_LAYER, AttentionScope.GLOBAL]:
            cross_layer_dependencies = self.cross_layer_analyzer.analyze_dependencies(attention_data)
        
        # Compute token attributions
        token_attributions = self._compute_token_attributions(attention_data, tokens)
        
        # Identify circuit components
        circuit_components = self._identify_circuit_components(attention_heads, attention_patterns)
        
        # Compute statistics
        pattern_statistics = self._compute_pattern_statistics(attention_patterns)
        
        # Prepare visualization data
        visualization_data = self._prepare_visualization_data(attention_heads, attention_patterns, cross_layer_dependencies)
        
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return AttentionAnalysisResult(
            attention_heads=attention_heads,
            attention_patterns=attention_patterns,
            cross_layer_dependencies=cross_layer_dependencies,
            token_attributions=token_attributions,
            circuit_components=circuit_components,
            pattern_statistics=pattern_statistics,
            visualization_data=visualization_data,
            execution_time_ms=execution_time_ms,
            metadata={
                "prompt": prompt,
                "num_tokens": len(tokens),
                "analyzed_layers": layers,
                "analysis_scope": scope.value
            }
        )
    
    def _create_attention_hooks(self, layers: List[int]) -> List[ActivationHookSpec]:
        """Create activation hooks for attention layers."""
        hooks = []
        for layer in layers:
            hook = ActivationHookSpec(
                layer_name=f"model.layers.{layer}.self_attn",
                component=ComponentType.ATTENTION,
                capture_input=True,
                capture_output=True
            )
            hooks.append(hook)
        return hooks
    
    def _capture_attention_activations(self, prompt: str, hooks: List[ActivationHookSpec]) -> Dict[str, Any]:
        """Capture attention activations during forward pass."""
        # This would integrate with the actual model's forward pass
        # For now, return mock data structure
        attention_data = {}
        
        # Mock attention data - in real implementation, this would come from hooks
        for hook in hooks:
            layer_num = int(hook.layer_name.split('.')[-2])
            # Create mock attention weights
            seq_len = 10  # Mock sequence length
            num_heads = 8  # Mock number of heads
            attention_weights = mx.random.uniform(0, 1, (num_heads, seq_len, seq_len))
            # Normalize to make it a proper attention distribution
            attention_weights = mx.softmax(attention_weights, axis=-1)
            attention_data[hook.layer_name] = attention_weights
        
        return attention_data
    
    def _analyze_attention_heads(self, attention_data: Dict[str, Any], tokens: List[int]) -> List[AttentionHead]:
        """Analyze individual attention heads."""
        attention_heads = []
        
        for layer_name, attention_weights in attention_data.items():
            layer_num = int(layer_name.split('.')[-2])
            num_heads = attention_weights.shape[0]
            
            for head_idx in range(num_heads):
                head_attention = attention_weights[head_idx]
                attention_head = self.pattern_classifier.classify_head_pattern(
                    head_attention, tokens, layer_num, head_idx
                )
                attention_heads.append(attention_head)
        
        return attention_heads
    
    def _identify_attention_patterns(self, attention_heads: List[AttentionHead]) -> List[AttentionPattern]:
        """Identify common attention patterns across heads."""
        patterns = []
        
        # Group heads by pattern type
        pattern_groups = {}
        for head in attention_heads:
            pattern_type = head.pattern_type
            if pattern_type not in pattern_groups:
                pattern_groups[pattern_type] = []
            pattern_groups[pattern_type].append(head)
        
        # Create pattern objects
        for pattern_type, heads in pattern_groups.items():
            if len(heads) >= 2:  # Only consider patterns with multiple heads
                strength = sum(head.confidence for head in heads) / len(heads)
                consistency = self._compute_pattern_consistency(heads)
                
                pattern = AttentionPattern(
                    pattern_type=pattern_type,
                    heads=heads,
                    strength=strength,
                    consistency=consistency,
                    token_positions=self._extract_token_positions(heads),
                    description=self._describe_pattern(pattern_type, heads),
                    examples=[]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _compute_pattern_consistency(self, heads: List[AttentionHead]) -> float:
        """Compute consistency of pattern across heads."""
        if len(heads) < 2:
            return 1.0
        
        confidences = [head.confidence for head in heads]
        mean_confidence = sum(confidences) / len(confidences)
        variance = sum((c - mean_confidence) ** 2 for c in confidences) / len(confidences)
        consistency = 1.0 / (1.0 + variance)
        return consistency
    
    def _extract_token_positions(self, heads: List[AttentionHead]) -> List[Tuple[int, int]]:
        """Extract common token positions from attention heads."""
        # Simplified implementation
        positions = []
        for head in heads[:3]:  # Limit to first 3 heads
            # Mock positions based on head metadata
            positions.append((0, 1))  # Mock source-target positions
        return positions
    
    def _describe_pattern(self, pattern_type: AttentionPatternType, heads: List[AttentionHead]) -> str:
        """Generate description for attention pattern."""
        descriptions = {
            AttentionPatternType.INDUCTION: f"Induction pattern with {len(heads)} heads copying information from previous occurrences",
            AttentionPatternType.PREVIOUS_TOKEN: f"Previous token pattern with {len(heads)} heads attending to immediately preceding tokens",
            AttentionPatternType.FIRST_TOKEN: f"First token pattern with {len(heads)} heads focusing on sequence beginning",
            AttentionPatternType.PUNCTUATION: f"Punctuation pattern with {len(heads)} heads attending to punctuation marks",
            AttentionPatternType.SYNTACTIC: f"Syntactic pattern with {len(heads)} heads processing grammatical relationships",
            AttentionPatternType.SEMANTIC: f"Semantic pattern with {len(heads)} heads processing meaning relationships"
        }
        return descriptions.get(pattern_type, f"Unknown pattern with {len(heads)} heads")
    
    def _compute_token_attributions(self, attention_data: Dict[str, Any], tokens: List[int]) -> Dict[int, Dict[str, float]]:
        """Compute importance attributions for each token."""
        attributions = {}
        
        for token_idx in range(len(tokens)):
            # Compute aggregate attention received by this token
            total_attention = 0.0
            attention_count = 0
            
            for layer_name, attention_weights in attention_data.items():
                if token_idx < attention_weights.shape[-1]:
                    token_attention = mx.sum(attention_weights[:, :, token_idx])
                    total_attention += float(token_attention)
                    attention_count += 1
            
            avg_attention = total_attention / max(attention_count, 1)
            
            attributions[token_idx] = {
                "importance": avg_attention,
                "role": self._determine_token_role(token_idx, tokens, avg_attention)
            }
        
        return attributions
    
    def _determine_token_role(self, token_idx: int, tokens: List[int], attention_score: float) -> str:
        """Determine the role of a token based on attention patterns."""
        if attention_score > 0.5:
            return "high_importance"
        elif attention_score > 0.2:
            return "medium_importance"
        else:
            return "low_importance"
    
    def _identify_circuit_components(self, attention_heads: List[AttentionHead], 
                                   attention_patterns: List[AttentionPattern]) -> List[Dict[str, Any]]:
        """Identify attention components that are part of neural circuits."""
        circuit_components = []
        
        for pattern in attention_patterns:
            if pattern.strength > 0.3:  # Threshold for circuit relevance
                component = {
                    "type": "attention_pattern",
                    "pattern_type": pattern.pattern_type.value,
                    "heads": [(head.layer, head.head) for head in pattern.heads],
                    "strength": pattern.strength,
                    "role": pattern.description,
                    "circuit_function": self._infer_circuit_function(pattern)
                }
                circuit_components.append(component)
        
        return circuit_components
    
    def _infer_circuit_function(self, pattern: AttentionPattern) -> str:
        """Infer the function of this pattern in neural circuits."""
        function_mapping = {
            AttentionPatternType.INDUCTION: "information_copying_and_completion",
            AttentionPatternType.PREVIOUS_TOKEN: "sequential_information_processing",
            AttentionPatternType.FIRST_TOKEN: "context_initialization_and_aggregation",
            AttentionPatternType.SYNTACTIC: "grammatical_structure_processing",
            AttentionPatternType.SEMANTIC: "meaning_extraction_and_association"
        }
        return function_mapping.get(pattern.pattern_type, "unknown_circuit_function")
    
    def _compute_pattern_statistics(self, attention_patterns: List[AttentionPattern]) -> Dict[str, Any]:
        """Compute statistics about discovered attention patterns."""
        if not attention_patterns:
            return {"total_patterns": 0}
        
        pattern_counts = {}
        total_strength = 0.0
        total_consistency = 0.0
        
        for pattern in attention_patterns:
            pattern_type = pattern.pattern_type.value
            pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
            total_strength += pattern.strength
            total_consistency += pattern.consistency
        
        return {
            "total_patterns": len(attention_patterns),
            "pattern_counts": pattern_counts,
            "average_strength": total_strength / len(attention_patterns),
            "average_consistency": total_consistency / len(attention_patterns),
            "strongest_pattern": max(attention_patterns, key=lambda p: p.strength).pattern_type.value,
            "most_consistent_pattern": max(attention_patterns, key=lambda p: p.consistency).pattern_type.value
        }
    
    def _prepare_visualization_data(self, attention_heads: List[AttentionHead], 
                                  attention_patterns: List[AttentionPattern],
                                  cross_layer_dependencies: List[CrossLayerDependency]) -> Dict[str, Any]:
        """Prepare data for attention visualization."""
        return {
            "attention_matrix": self._create_attention_matrix(attention_heads),
            "pattern_graph": self._create_pattern_graph(attention_patterns),
            "dependency_graph": self._create_dependency_graph(cross_layer_dependencies),
            "head_classifications": self._create_head_classification_data(attention_heads)
        }
    
    def _create_attention_matrix(self, attention_heads: List[AttentionHead]) -> Dict[str, Any]:
        """Create attention matrix for visualization."""
        # Group heads by layer
        layer_data = {}
        for head in attention_heads:
            layer = head.layer
            if layer not in layer_data:
                layer_data[layer] = []
            layer_data[layer].append({
                "head": head.head,
                "pattern_type": head.pattern_type.value,
                "confidence": head.confidence,
                "attention_weights": head.attention_weights.tolist() if hasattr(head.attention_weights, 'tolist') else []
            })
        
        return layer_data
    
    def _create_pattern_graph(self, attention_patterns: List[AttentionPattern]) -> Dict[str, Any]:
        """Create pattern graph for visualization."""
        nodes = []
        edges = []
        
        for i, pattern in enumerate(attention_patterns):
            nodes.append({
                "id": f"pattern_{i}",
                "type": pattern.pattern_type.value,
                "strength": pattern.strength,
                "consistency": pattern.consistency,
                "head_count": len(pattern.heads)
            })
        
        return {"nodes": nodes, "edges": edges}
    
    def _create_dependency_graph(self, dependencies: List[CrossLayerDependency]) -> Dict[str, Any]:
        """Create dependency graph for visualization."""
        nodes = set()
        edges = []
        
        for dep in dependencies:
            nodes.add(dep.source_layer)
            nodes.add(dep.target_layer)
            edges.append({
                "source": dep.source_layer,
                "target": dep.target_layer,
                "type": dep.dependency_type,
                "strength": dep.strength
            })
        
        node_list = [{"id": layer, "type": "attention_layer"} for layer in sorted(nodes)]
        return {"nodes": node_list, "edges": edges}
    
    def _create_head_classification_data(self, attention_heads: List[AttentionHead]) -> Dict[str, Any]:
        """Create head classification data for visualization."""
        classifications = {}
        for head in attention_heads:
            layer_key = f"layer_{head.layer}"
            if layer_key not in classifications:
                classifications[layer_key] = []
            
            classifications[layer_key].append({
                "head": head.head,
                "pattern_type": head.pattern_type.value,
                "confidence": head.confidence,
                "circuit_role": head.circuit_role
            })
        
        return classifications

def create_attention_analysis_pipeline(model: nn.Module) -> AttentionAnalyzer:
    """Factory function to create an attention analysis pipeline."""
    return AttentionAnalyzer(model)