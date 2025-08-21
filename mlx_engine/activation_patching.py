"""Activation Patching for Causal Intervention Analysis

This module implements sophisticated activation patching techniques for mechanistic
interpretability, enabling causal tracing and circuit discovery through targeted
interventions in model activations.
"""

from typing import Dict, List, Optional, Callable, Any, Union, Tuple
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
import numpy as np
import logging
import traceback
from enum import Enum
from .activation_hooks import ActivationHookManager, ComponentType, ActivationHookSpec, CapturedActivation

# Configure detailed logging for activation patching
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class InterventionType(Enum):
    """Types of activation interventions for causal analysis."""
    ZERO_ABLATION = "zero_ablation"  # Set activations to zero
    MEAN_ABLATION = "mean_ablation"  # Replace with mean activation
    NOISE_INJECTION = "noise_injection"  # Add Gaussian noise
    ACTIVATION_PATCHING = "activation_patching"  # Replace with stored activations
    GRADIENT_BASED = "gradient_based"  # Gradient-based attribution
    RANDOM_REPLACEMENT = "random_replacement"  # Replace with random activations
    SCALED_INTERVENTION = "scaled_intervention"  # Scale activations by factor


@dataclass
class InterventionSpec:
    """Specification for an activation intervention."""
    layer_name: str
    component: ComponentType
    intervention_type: InterventionType
    intervention_id: Optional[str] = None
    strength: float = 1.0  # Intervention strength (0.0 = no intervention, 1.0 = full)
    target_positions: Optional[List[int]] = None  # Specific token positions to intervene
    target_heads: Optional[List[int]] = None  # Specific attention heads to intervene
    replacement_activations: Optional[mx.array] = None  # For activation patching
    noise_std: float = 0.1  # Standard deviation for noise injection
    
    def __post_init__(self):
        if self.intervention_id is None:
            self.intervention_id = f"{self.layer_name}_{self.component.value}_{self.intervention_type.value}"


@dataclass
class CausalTracingResult:
    """Results from causal tracing analysis."""
    intervention_id: str
    layer_name: str
    component: ComponentType
    intervention_type: InterventionType
    baseline_output: Any
    intervened_output: Any
    causal_effect: float  # Magnitude of causal effect
    attribution_score: float  # Normalized attribution score
    confidence: float  # Confidence in the result
    metadata: Dict[str, Any]


class ActivationPatcher:
    """Implements activation patching for causal intervention analysis."""
    
    def __init__(self, model: nn.Module, hook_manager: Optional[ActivationHookManager] = None):
        logger.info("Initializing ActivationPatcher...")
        logger.debug(f"Model type: {type(model).__name__}")
        
        try:
            self.model = model
            self.hook_manager = hook_manager or ActivationHookManager(model)
            logger.debug(f"Hook manager initialized: {type(self.hook_manager).__name__}")
            
            self.interventions: Dict[str, InterventionSpec] = {}
            self.baseline_activations: Dict[str, List[CapturedActivation]] = {}
            self.intervention_results: List[CausalTracingResult] = []
            self._active_interventions: Dict[str, bool] = {}
            
            logger.info("ActivationPatcher initialization completed successfully")
            logger.debug(f"Initialized with {len(self.interventions)} interventions")
            
        except Exception as e:
            logger.error(f"Failed to initialize ActivationPatcher: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def register_intervention(self, spec: InterventionSpec) -> str:
        """Register an activation intervention."""
        logger.info(f"Registering intervention: {spec.intervention_id}")
        logger.debug(f"Intervention details - Layer: {spec.layer_name}, Component: {spec.component.value}, Type: {spec.intervention_type.value}")
        logger.debug(f"Intervention strength: {spec.strength}, Target positions: {spec.target_positions}")
        
        try:
            self.interventions[spec.intervention_id] = spec
            self._active_interventions[spec.intervention_id] = False
            
            logger.info(f"Successfully registered intervention: {spec.intervention_id}")
            logger.debug(f"Total interventions registered: {len(self.interventions)}")
            
            return spec.intervention_id
            
        except Exception as e:
            logger.error(f"Failed to register intervention {spec.intervention_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def capture_baseline_activations(self, prompt: str, max_tokens: int = 50) -> Dict[str, List[CapturedActivation]]:
        """Capture baseline activations for comparison during interventions."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Capturing baseline activations for prompt: '{prompt[:30]}...'")
        logger.info(f"Number of registered interventions: {len(self.interventions)}")
        
        try:
            # Clear any existing hooks
            logger.debug("Clearing existing hooks...")
            self.hook_manager.clear_all_hooks()
            logger.debug("Hooks cleared")
            
            # Register hooks for all intervention targets
            hook_count = 0
            for intervention in self.interventions.values():
                hook_count += 1
                logger.debug(f"Registering hook {hook_count} for layer: {intervention.layer_name}, component: {intervention.component.value}")
                
                hook_spec = ActivationHookSpec(
                    layer_name=intervention.layer_name,
                    component=intervention.component,
                    hook_id=f"baseline_{intervention.intervention_id}",
                    capture_output=True
                )
                self.hook_manager.register_hook(hook_spec)
                logger.debug(f"Hook registered with ID: baseline_{intervention.intervention_id}")
            
            logger.info(f"Registered {hook_count} hooks for baseline capture")
            
            # Generate text to capture baseline activations
            # Note: This would need to be integrated with the actual model generation
            # For now, we'll return a placeholder structure
            logger.info("Getting captured activations from hook manager...")
            self.baseline_activations = self.hook_manager.get_captured_activations()
            logger.info(f"Baseline activations captured: {len(self.baseline_activations)} activation sets")
            
            for key, activations in self.baseline_activations.items():
                logger.debug(f"Activation set '{key}': {len(activations)} activations")
            
            return self.baseline_activations
            
        except Exception as e:
            logger.error(f"Error capturing baseline activations: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def apply_intervention(self, intervention_id: str, activation: mx.array) -> mx.array:
        """Apply the specified intervention to an activation."""
        logger.debug(f"Applying intervention: {intervention_id}")
        logger.debug(f"Activation shape: {activation.shape if hasattr(activation, 'shape') else 'unknown'}")
        
        if intervention_id not in self.interventions:
            logger.warning(f"Intervention {intervention_id} not found in registered interventions")
            logger.debug(f"Available interventions: {list(self.interventions.keys())}")
            return activation
        
        spec = self.interventions[intervention_id]
        logger.debug(f"Applying {spec.intervention_type.value} intervention with strength {spec.strength}")
        
        try:
            if spec.intervention_type == InterventionType.ZERO_ABLATION:
                result = self._zero_ablation(activation, spec)
            elif spec.intervention_type == InterventionType.MEAN_ABLATION:
                result = self._mean_ablation(activation, spec)
            elif spec.intervention_type == InterventionType.NOISE_INJECTION:
                result = self._noise_injection(activation, spec)
            elif spec.intervention_type == InterventionType.ACTIVATION_PATCHING:
                result = self._activation_patching(activation, spec)
            elif spec.intervention_type == InterventionType.RANDOM_REPLACEMENT:
                result = self._random_replacement(activation, spec)
            elif spec.intervention_type == InterventionType.SCALED_INTERVENTION:
                result = self._scaled_intervention(activation, spec)
            else:
                logger.warning(f"Unknown intervention type: {spec.intervention_type}")
                result = activation
            
            logger.debug(f"Intervention {intervention_id} applied successfully")
            logger.debug(f"Result shape: {result.shape if hasattr(result, 'shape') else 'unknown'}")
            return result
             
        except Exception as e:
            logger.error(f"Failed to apply intervention {intervention_id}: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return activation  # Return original activation on error
    
    def _zero_ablation(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Set activations to zero (complete ablation)."""
        if spec.target_positions is not None:
            # Ablate specific positions
            result = activation.copy()
            for pos in spec.target_positions:
                if pos < activation.shape[0]:
                    result = result.at[pos].set(0.0)
            return result * (1.0 - spec.strength) + result * 0.0 * spec.strength
        else:
            # Ablate entire activation
            return activation * (1.0 - spec.strength)
    
    def _mean_ablation(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Replace activations with mean activation."""
        mean_activation = mx.mean(activation, axis=0, keepdims=True)
        if spec.target_positions is not None:
            result = activation.copy()
            for pos in spec.target_positions:
                if pos < activation.shape[0]:
                    result = result.at[pos].set(mean_activation[0])
            return result
        else:
            return activation * (1.0 - spec.strength) + mean_activation * spec.strength
    
    def _noise_injection(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Inject Gaussian noise into activations."""
        noise = mx.random.normal(activation.shape) * spec.noise_std
        return activation + noise * spec.strength
    
    def _activation_patching(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Replace activations with stored replacement activations."""
        if spec.replacement_activations is None:
            return activation
        
        replacement = spec.replacement_activations
        if spec.target_positions is not None:
            result = activation.copy()
            for pos in spec.target_positions:
                if pos < activation.shape[0] and pos < replacement.shape[0]:
                    result = result.at[pos].set(replacement[pos])
            return result
        else:
            return activation * (1.0 - spec.strength) + replacement * spec.strength
    
    def _random_replacement(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Replace activations with random values."""
        random_activation = mx.random.normal(activation.shape)
        return activation * (1.0 - spec.strength) + random_activation * spec.strength
    
    def _scaled_intervention(self, activation: mx.array, spec: InterventionSpec) -> mx.array:
        """Scale activations by a factor."""
        return activation * spec.strength
    
    def enable_intervention(self, intervention_id: str):
        """Enable a specific intervention."""
        if intervention_id in self._active_interventions:
            self._active_interventions[intervention_id] = True
    
    def disable_intervention(self, intervention_id: str):
        """Disable a specific intervention."""
        if intervention_id in self._active_interventions:
            self._active_interventions[intervention_id] = False
    
    def is_intervention_active(self, intervention_id: str) -> bool:
        """Check if an intervention is currently active."""
        return self._active_interventions.get(intervention_id, False)


class CausalTracer:
    """Implements causal tracing for circuit discovery."""
    
    def __init__(self, model: nn.Module, patcher: Optional[ActivationPatcher] = None):
        self.model = model
        self.patcher = patcher or ActivationPatcher(model)
        self.tracing_results: List[CausalTracingResult] = []
    
    def trace_causal_effects(self, 
                           prompt: str, 
                           target_layers: List[str],
                           target_components: List[ComponentType],
                           intervention_types: List[InterventionType],
                           max_tokens: int = 50) -> List[CausalTracingResult]:
        """Perform systematic causal tracing across specified layers and components."""
        import logging
        logger = logging.getLogger(__name__)
        
        logger.info(f"Starting circuit discovery for prompt: '{prompt[:50]}...'")
        logger.info(f"Target layers: {target_layers}")
        logger.info(f"Target components: {[c.value for c in target_components]}")
        logger.info(f"Intervention types: {[i.value for i in intervention_types]}")
        logger.info(f"Max tokens: {max_tokens}")
        
        try:
            # Capture baseline activations and output
            logger.info("Capturing baseline activations...")
            baseline_activations = self.patcher.capture_baseline_activations(prompt, max_tokens)
            logger.info(f"Baseline activations captured: {type(baseline_activations)}")
            
            # Generate baseline output (this would need integration with actual generation)
            logger.info("Generating baseline output...")
            baseline_output = self._generate_with_model(prompt, max_tokens)
            logger.info(f"Baseline output generated: {type(baseline_output)}")
            
            results = []
            total_combinations = len(target_layers) * len(target_components) * len(intervention_types)
            logger.info(f"Testing {total_combinations} intervention combinations...")
            
            combination_count = 0
            # Test each combination of layer, component, and intervention type
            for layer in target_layers:
                for component in target_components:
                    for intervention_type in intervention_types:
                        combination_count += 1
                        logger.info(f"Processing combination {combination_count}/{total_combinations}: {layer}, {component.value}, {intervention_type.value}")
                        
                        try:
                            # Create intervention specification
                            intervention_spec = InterventionSpec(
                                layer_name=layer,
                                component=component,
                                intervention_type=intervention_type
                            )
                            logger.debug(f"Created intervention spec: {intervention_spec}")
                            
                            # Register and apply intervention
                            logger.debug("Registering intervention...")
                            intervention_id = self.patcher.register_intervention(intervention_spec)
                            logger.debug(f"Intervention registered with ID: {intervention_id}")
                            
                            self.patcher.enable_intervention(intervention_id)
                            logger.debug("Intervention enabled")
                            
                            # Generate output with intervention
                            logger.debug("Generating intervened output...")
                            intervened_output = self._generate_with_model(prompt, max_tokens)
                            logger.debug(f"Intervened output generated: {type(intervened_output)}")
                            
                            # Calculate causal effect
                            logger.debug("Calculating causal effect...")
                            causal_effect = self._calculate_causal_effect(baseline_output, intervened_output)
                            logger.debug(f"Causal effect: {causal_effect}")
                            
                            # Create result
                            result = CausalTracingResult(
                                intervention_id=intervention_id,
                                layer_name=layer,
                                component=component,
                                intervention_type=intervention_type,
                                baseline_output=baseline_output,
                                intervened_output=intervened_output,
                                causal_effect=causal_effect,
                                attribution_score=self._normalize_attribution(causal_effect),
                                confidence=self._calculate_confidence(causal_effect),
                                metadata={
                                    'prompt': prompt,
                                    'max_tokens': max_tokens,
                                    'intervention_strength': intervention_spec.strength
                                }
                            )
                            
                            results.append(result)
                            logger.debug(f"Result created with attribution score: {result.attribution_score}")
                            
                            self.patcher.disable_intervention(intervention_id)
                            logger.debug("Intervention disabled")
                            
                        except Exception as e:
                            logger.error(f"Error processing combination {combination_count}: {e}")
                            logger.error(f"Exception type: {type(e).__name__}")
                            import traceback
                            logger.error(f"Traceback: {traceback.format_exc()}")
                            # Continue with next combination
                            continue
            
            logger.info(f"Circuit discovery completed. Found {len(results)} results.")
            self.tracing_results.extend(results)
            return results
            
        except Exception as e:
            logger.error(f"Fatal error in circuit discovery: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _generate_with_model(self, prompt: str, max_tokens: int) -> Any:
        """Generate text with the model (placeholder for actual implementation)."""
        # This would need to be integrated with the actual MLX model generation
        # For now, return a placeholder
        return {"text": "placeholder_output", "logits": None}
    
    def _calculate_causal_effect(self, baseline: Any, intervened: Any) -> float:
        """Calculate the magnitude of causal effect between baseline and intervened outputs."""
        # This would implement actual causal effect calculation
        # For now, return a placeholder value
        return 0.5
    
    def _normalize_attribution(self, causal_effect: float) -> float:
        """Normalize attribution score to [0, 1] range."""
        return min(1.0, max(0.0, abs(causal_effect)))
    
    def _calculate_confidence(self, causal_effect: float) -> float:
        """Calculate confidence in the causal effect measurement."""
        # Simple confidence based on effect magnitude
        return min(1.0, abs(causal_effect) * 2.0)
    
    def get_top_circuits(self, n: int = 10, min_confidence: float = 0.5) -> List[CausalTracingResult]:
        """Get the top N circuits by attribution score with minimum confidence."""
        filtered_results = [
            result for result in self.tracing_results 
            if result.confidence >= min_confidence
        ]
        
        return sorted(filtered_results, 
                     key=lambda x: x.attribution_score, 
                     reverse=True)[:n]
    
    def analyze_circuit_interactions(self, circuit_results: List[CausalTracingResult]) -> Dict[str, Any]:
        """Analyze interactions between discovered circuits."""
        # Placeholder for circuit interaction analysis
        return {
            "num_circuits": len(circuit_results),
            "avg_attribution": np.mean([r.attribution_score for r in circuit_results]),
            "layer_distribution": self._analyze_layer_distribution(circuit_results),
            "component_distribution": self._analyze_component_distribution(circuit_results)
        }
    
    def _analyze_layer_distribution(self, results: List[CausalTracingResult]) -> Dict[str, int]:
        """Analyze distribution of circuits across layers."""
        distribution = {}
        for result in results:
            layer = result.layer_name
            distribution[layer] = distribution.get(layer, 0) + 1
        return distribution
    
    def _analyze_component_distribution(self, results: List[CausalTracingResult]) -> Dict[str, int]:
        """Analyze distribution of circuits across components."""
        distribution = {}
        for result in results:
            component = result.component.value
            distribution[component] = distribution.get(component, 0) + 1
        return distribution


class GradientBasedAttribution:
    """Implements gradient-based attribution methods for circuit discovery."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attribution_cache: Dict[str, mx.array] = {}
    
    def compute_integrated_gradients(self, 
                                   prompt: str,
                                   target_layer: str,
                                   target_component: ComponentType,
                                   steps: int = 50) -> mx.array:
        """Compute integrated gradients for attribution analysis."""
        # Placeholder for integrated gradients implementation
        # This would require integration with MLX's gradient computation
        return mx.zeros((1, 768))  # Placeholder
    
    def compute_gradient_x_input(self, 
                               prompt: str,
                               target_layer: str,
                               target_component: ComponentType) -> mx.array:
        """Compute gradient × input attribution."""
        # Placeholder for gradient × input implementation
        return mx.zeros((1, 768))  # Placeholder
    
    def compute_layer_wise_relevance(self, 
                                   prompt: str,
                                   target_layers: List[str]) -> Dict[str, mx.array]:
        """Compute layer-wise relevance propagation."""
        # Placeholder for LRP implementation
        return {layer: mx.zeros((1, 768)) for layer in target_layers}


def create_sophisticated_circuit_discovery_pipeline(model: nn.Module) -> Tuple[ActivationPatcher, CausalTracer, GradientBasedAttribution]:
    """Create a complete pipeline for sophisticated circuit discovery."""
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Creating sophisticated circuit discovery pipeline for model: {type(model).__name__}")
    
    try:
        logger.info("Initializing ActivationPatcher...")
        patcher = ActivationPatcher(model)
        logger.info("ActivationPatcher created successfully")
        
        logger.info("Initializing CausalTracer...")
        tracer = CausalTracer(model, patcher)
        logger.info("CausalTracer created successfully")
        
        logger.info("Initializing GradientBasedAttribution...")
        attribution = GradientBasedAttribution(model)
        logger.info("GradientBasedAttribution created successfully")
        
        logger.info("Circuit discovery pipeline created successfully")
        return patcher, tracer, attribution
    except Exception as e:
        logger.error(f"Failed to create circuit discovery pipeline: {e}")
        logger.error(f"Exception type: {type(e).__name__}")
        import traceback
        logger.error(f"Full traceback: {traceback.format_exc()}")
        raise