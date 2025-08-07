"""
Activation Hook Infrastructure for NeuroScope Integration

This module provides the core infrastructure for capturing internal activations
from MLX models during inference, enabling mechanistic interpretability analysis
through NeuroScope.
"""

from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass
import mlx.core as mx
import mlx.nn as nn
from enum import Enum


class ComponentType(Enum):
    """Types of model components that can be hooked for activation capture."""
    RESIDUAL = "residual"
    ATTENTION = "attention"
    MLP = "mlp"
    EMBEDDING = "embedding"
    LAYERNORM = "layernorm"
    ATTENTION_SCORES = "attention_scores"
    ATTENTION_PATTERN = "attention_pattern"
    KEY = "key"
    QUERY = "query"
    VALUE = "value"


@dataclass
class ActivationHookSpec:
    """Specification for an activation hook."""
    layer_name: str
    component: ComponentType
    hook_id: Optional[str] = None
    capture_input: bool = False
    capture_output: bool = True
    
    def __post_init__(self):
        if self.hook_id is None:
            self.hook_id = f"{self.layer_name}_{self.component.value}"


@dataclass
class CapturedActivation:
    """Container for captured activation data."""
    hook_id: str
    layer_name: str
    component: ComponentType
    activation: mx.array
    shape: tuple
    dtype: str
    is_input: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'hook_id': self.hook_id,
            'layer_name': self.layer_name,
            'component': self.component.value,
            'shape': list(self.shape),
            'dtype': self.dtype,
            'is_input': self.is_input,
            # Note: actual activation data serialized separately for efficiency
        }


class ActivationHook:
    """Hook for capturing activations from model components."""
    
    def __init__(self, spec: ActivationHookSpec):
        self.spec = spec
        self.activations: List[CapturedActivation] = []
        self._enabled = True
    
    def __call__(self, module: nn.Module, input_data: Union[mx.array, tuple], output_data: mx.array) -> mx.array:
        """Hook function called during forward pass."""
        if not self._enabled:
            return output_data
            
        # Capture input if requested
        if self.spec.capture_input:
            input_array = input_data if isinstance(input_data, mx.array) else input_data[0]
            self.activations.append(CapturedActivation(
                hook_id=self.spec.hook_id,
                layer_name=self.spec.layer_name,
                component=self.spec.component,
                activation=input_array.copy(),
                shape=input_array.shape,
                dtype=str(input_array.dtype),
                is_input=True
            ))
        
        # Capture output if requested
        if self.spec.capture_output:
            self.activations.append(CapturedActivation(
                hook_id=self.spec.hook_id,
                layer_name=self.spec.layer_name,
                component=self.spec.component,
                activation=output_data.copy(),
                shape=output_data.shape,
                dtype=str(output_data.dtype),
                is_input=False
            ))
        
        return output_data
    
    def clear(self):
        """Clear captured activations."""
        self.activations.clear()
    
    def enable(self):
        """Enable activation capture."""
        self._enabled = True
    
    def disable(self):
        """Disable activation capture."""
        self._enabled = False


class ActivationHookManager:
    """Manages activation hooks for a model."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.hooks: Dict[str, ActivationHook] = {}
        self._registered_handles: Dict[str, Any] = {}
    
    def register_hook(self, spec: ActivationHookSpec) -> str:
        """Register an activation hook on the model."""
        hook = ActivationHook(spec)
        self.hooks[spec.hook_id] = hook
        
        # Find the target module
        target_module = self._find_module(spec.layer_name, spec.component)
        if target_module is None:
            raise ValueError(f"Could not find module for {spec.layer_name}.{spec.component.value}")
        
        # Register the hook
        handle = target_module.register_forward_hook(hook)
        self._registered_handles[spec.hook_id] = handle
        
        return spec.hook_id
    
    def unregister_hook(self, hook_id: str):
        """Unregister an activation hook."""
        if hook_id in self._registered_handles:
            self._registered_handles[hook_id].remove()
            del self._registered_handles[hook_id]
        
        if hook_id in self.hooks:
            del self.hooks[hook_id]
    
    def clear_all_hooks(self):
        """Clear all registered hooks."""
        for hook_id in list(self.hooks.keys()):
            self.unregister_hook(hook_id)
    
    def get_activations(self, hook_id: Optional[str] = None) -> Dict[str, List[CapturedActivation]]:
        """Get captured activations from hooks."""
        if hook_id:
            if hook_id in self.hooks:
                return {hook_id: self.hooks[hook_id].activations}
            else:
                return {}
        
        return {hid: hook.activations for hid, hook in self.hooks.items()}
    
    def clear_activations(self, hook_id: Optional[str] = None):
        """Clear captured activations."""
        if hook_id:
            if hook_id in self.hooks:
                self.hooks[hook_id].clear()
        else:
            for hook in self.hooks.values():
                hook.clear()
    
    def _find_module(self, layer_name: str, component: ComponentType) -> Optional[nn.Module]:
        """Find the target module for hooking based on layer name and component type."""
        # This is a simplified implementation - in practice, you'd need to handle
        # different model architectures and their specific naming conventions
        
        try:
            # Handle transformer layer access patterns
            if layer_name.startswith('transformer.h.'):
                layer_idx = int(layer_name.split('.')[-1])
                layer = self.model.transformer.h[layer_idx]
                
                if component == ComponentType.RESIDUAL:
                    return layer
                elif component == ComponentType.ATTENTION:
                    return layer.attn
                elif component == ComponentType.MLP:
                    return layer.mlp
                elif component == ComponentType.LAYERNORM:
                    return layer.ln_1  # or ln_2 depending on position
                
            # Handle embedding layer
            elif layer_name == 'transformer.wte':
                return self.model.transformer.wte
            
            # Handle other patterns as needed
            else:
                # Generic attribute access
                parts = layer_name.split('.')
                module = self.model
                for part in parts:
                    if part.isdigit():
                        module = module[int(part)]
                    else:
                        module = getattr(module, part)
                return module
                
        except (AttributeError, IndexError, ValueError):
            return None
        
        return None


def serialize_activations(activations: Dict[str, List[CapturedActivation]], 
                         format: str = 'numpy') -> Dict[str, Any]:
    """Serialize captured activations for transmission."""
    serialized = {}
    
    for hook_id, activation_list in activations.items():
        serialized[hook_id] = []
        
        for activation in activation_list:
            activation_data = activation.to_dict()
            
            # Serialize the actual tensor data
            if format == 'numpy':
                import numpy as np
                activation_data['data'] = np.array(activation.activation).tolist()
            elif format == 'base64':
                import base64
                import numpy as np
                arr = np.array(activation.activation)
                activation_data['data'] = base64.b64encode(arr.tobytes()).decode('utf-8')
            else:
                # Just include metadata without data for efficiency
                activation_data['data'] = None
            
            serialized[hook_id].append(activation_data)
    
    return serialized