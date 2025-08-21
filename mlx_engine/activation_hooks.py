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
from typing import Any, Optional, Union, List, Tuple, Dict, Callable


class ExpertWrapper(nn.Module):
    """Wrapper class to access a specific expert in a MoE layer.
    
    This wrapper allows us to hook into individual experts in the MoE layer
    by providing a callable interface that routes inputs to the specified expert.
    """
    
    def __init__(self, experts: Any, expert_idx: int):
        """Initialize the ExpertWrapper.
        
        Args:
            experts: The experts module from the MoE layer
            expert_idx: Index of the expert to wrap
        """
        super().__init__()
        self.experts = experts
        self.expert_idx = expert_idx
    
    def __call__(self, x: mx.array) -> mx.array:
        """Call the wrapped expert with the input.
        
        Args:
            x: Input tensor
            
        Returns:
            Output from the expert
        """
        # Create indices to select the specified expert for all tokens
        batch_size = x.shape[0]
        expert_indices = mx.full((batch_size, 1), self.expert_idx)
        
        # Call the experts module with the selected expert
        return self.experts(x, expert_indices)


class ComponentType(Enum):
    """Types of model components that can be hooked for activation capture."""
    RESIDUAL = "residual"
    ATTENTION = "attention"
    MLP = "mlp"
    MOE = "moe"  # For Mixture of Experts layers
    EXPERT = "expert"  # Individual experts in MoE
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
        print(f"\n[DEBUG] ActivationHook.__call__ called for {self.spec.hook_id}")
        print(f"[DEBUG] Hook spec: {self.spec}")
        print(f"[DEBUG] Module type: {type(module).__name__ if module is not None else 'None'}")
        print(f"[DEBUG] Input type: {type(input_data)}, Output type: {type(output_data)}")
        
        if not self._enabled:
            print("[DEBUG] Hook is disabled, skipping capture")
            return output_data
            
        # Capture input if requested
        if self.spec.capture_input:
            try:
                input_array = input_data if isinstance(input_data, mx.array) else (input_data[0] if input_data and len(input_data) > 0 else None)
                if input_array is not None:
                    print(f"[DEBUG] Capturing input for {self.spec.hook_id}, shape: {input_array.shape}, dtype: {input_array.dtype}")
                    self.activations.append(CapturedActivation(
                        hook_id=self.spec.hook_id,
                        layer_name=self.spec.layer_name,
                        component=self.spec.component,
                        activation=input_array.copy(),
                        shape=input_array.shape,
                        dtype=str(input_array.dtype),
                        is_input=True
                    ))
                    print(f"[DEBUG] Successfully captured input for {self.spec.hook_id}")
                else:
                    print(f"[WARNING] Input data is None for {self.spec.hook_id}")
            except Exception as e:
                print(f"[ERROR] Failed to capture input for {self.spec.hook_id}: {e}")
        
        # Capture output if requested
        if self.spec.capture_output:
            try:
                if output_data is not None:
                    print(f"[DEBUG] Capturing output for {self.spec.hook_id}, shape: {output_data.shape}, dtype: {output_data.dtype}")
                    self.activations.append(CapturedActivation(
                        hook_id=self.spec.hook_id,
                        layer_name=self.spec.layer_name,
                        component=self.spec.component,
                        activation=output_data.copy(),
                        shape=output_data.shape,
                        dtype=str(output_data.dtype),
                        is_input=False
                    ))
                    print(f"[DEBUG] Successfully captured output for {self.spec.hook_id}")
                else:
                    print(f"[WARNING] Output data is None for {self.spec.hook_id}")
            except Exception as e:
                print(f"[ERROR] Failed to capture output for {self.spec.hook_id}: {e}")
        
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
    
    def register_hook(self, spec: ActivationHookSpec) -> bool:
        """Register an activation hook on the model.
        
        This method has been updated to handle MLX models that don't support PyTorch-style hooks.
        For MLX models, we'll store the hook and the target module, and the actual hooking
        will be handled by the model's forward pass.
        """
        print(f"\n[DEBUG] ====== REGISTERING HOOK ======")
        print(f"[DEBUG] Hook spec: {spec}")
        print("=============================")
        
        if not spec.layer_name or not spec.hook_id:
            print(f"[ERROR] Invalid hook registration: layer_name and hook_id are required")
            return False
            
        if not spec.capture_input and not spec.capture_output:
            print(f"[WARNING] Hook {spec.hook_id} won't capture anything: both capture_input and capture_output are False")
            return False
            
        # Create the hook
        hook = ActivationHook(spec)
        self.hooks[spec.hook_id] = hook
        
        # Find the target module
        component = spec.component.value if hasattr(spec.component, 'value') else spec.component
        print(f"[DEBUG] Searching for module: {spec.layer_name} with component: {component}")
        
        target_module = self._find_module(spec.layer_name, spec.component)
        if target_module is None:
            print(f"[ERROR] Failed to find module for {spec.layer_name} with component {component}")
            return False
            
        print(f"[DEBUG] Found target module: {type(target_module).__name__}")
        print(f"[DEBUG] Target module type: {type(target_module).__name__}")
        print(f"[DEBUG] Target module has {len([attr for attr in dir(target_module) if not attr.startswith('_')])} attributes")
        
        # Check if the module is callable
        is_callable = hasattr(target_module, '__call__')
        print(f"[DEBUG] Module is callable: {is_callable}")
        if is_callable:
            print(f"[DEBUG] Module __call__ method available")
            
        # Check if the module has a forward method
        has_forward = hasattr(target_module, 'forward')
        print(f"[DEBUG] Module has forward method: {has_forward}")
            
        # Check if the module is an instance of nn.Module
        is_nn_module = isinstance(target_module, nn.Module)
        print(f"[DEBUG] Module is nn.Module: {is_nn_module}")
        
        # Check if the module is an MLX model
        is_mlx_model = hasattr(target_module, 'mx') or hasattr(target_module, 'mlx')
        print(f"[DEBUG] Module is MLX model: {is_mlx_model}")
        
        # Store the hook and target module
        # For MLX models, we'll need to manually call the hook during the forward pass
        self._registered_handles[spec.hook_id] = {
            'module': target_module,
            'hook': hook,
            'original_forward': None
        }
        
        # For MLX models, we'll need to patch the forward method
        # to call our hook before and after the original forward
        if hasattr(target_module, '__call__') and not hasattr(target_module, 'register_forward_hook'):
            print(f"[DEBUG] Patching __call__ method for MLX model at {spec.layer_name}.{spec.component}")
            print(f"[DEBUG] Target module type: {type(target_module).__name__}")
            print(f"[DEBUG] Target module ID: {id(target_module)}")
            
            original_call = target_module.__call__
            
            def patched_call(*args, **kwargs):
                hook_id = spec.hook_id
                # Call the hook with input if needed
                if spec.capture_input:
                    input_data = args[0] if args else None
                    print(f"[DEBUG] [Hook {hook_id}] Input hook triggered at {spec.layer_name}.{spec.component}")
                    print(f"[DEBUG] [Hook {hook_id}] Input type: {type(input_data).__name__}")
                    if hasattr(input_data, 'shape'):
                        print(f"[DEBUG] [Hook {hook_id}] Input shape: {input_data.shape}")
                    hook(None, input_data, None)
                
                # Call the original forward
                print(f"[DEBUG] [Hook {hook_id}] Calling original __call__ at {spec.layer_name}.{spec.component}")
                try:
                    output = original_call(*args, **kwargs)
                    print(f"[DEBUG] [Hook {hook_id}] Original __call__ completed successfully")
                except Exception as e:
                    print(f"[ERROR] [Hook {hook_id}] Error in original __call__: {e}")
                    import traceback
                    traceback.print_exc()
                    raise
                
                # Call the hook with output if needed
                if spec.capture_output:
                    print(f"[DEBUG] [Hook {hook_id}] Output hook triggered at {spec.layer_name}.{spec.component}")
                    print(f"[DEBUG] [Hook {hook_id}] Output type: {type(output).__name__}")
                    if hasattr(output, 'shape'):
                        print(f"[DEBUG] [Hook {hook_id}] Output shape: {output.shape}")
                    hook(None, None, output)
                
                return output
            
            # Store the original call for cleanup
            self._registered_handles[spec.hook_id]['original_call'] = original_call
            
            # Patch the __call__ method
            target_module.__call__ = patched_call
            print(f"[DEBUG] Successfully patched __call__ for {spec.hook_id}")
            print(f"[DEBUG] New __call__ method ID: {id(target_module.__call__)}")
            print(f"[DEBUG] Original __call__ method ID: {id(original_call) if original_call else 'N/A'}")
        elif hasattr(target_module, 'register_forward_hook'):
            # Standard PyTorch-style hook registration
            print("[DEBUG] Registering PyTorch-style forward hook")
            try:
                handle = target_module.register_forward_hook(hook)
                self._registered_handles[spec.hook_id]['handle'] = handle
                print(f"[DEBUG] Successfully registered PyTorch hook for {spec.hook_id}")
            except Exception as e:
                print(f"[ERROR] Failed to register PyTorch hook: {e}")
                raise
        else:
            error_msg = f"Module {spec.layer_name}.{spec.component.value} does not support hooking"
            print(f"[WARNING] {error_msg}")
            
        return spec.hook_id

    def add_hook(self, spec: Union[ActivationHookSpec, dict]) -> str:
        """Alias for register_hook to maintain compatibility with upstream code."""
        # Convert dict to ActivationHookSpec if needed
        if isinstance(spec, dict):
            spec = ActivationHookSpec(
                layer_name=spec['layer_name'],
                component=ComponentType(spec.get('component', 'attention')),
                hook_id=spec.get('hook_id'),
                capture_input=spec.get('capture_input', False),
                capture_output=spec.get('capture_output', True)
            )
        return self.register_hook(spec)

    def enable_hooks(self):
        """Enable all registered hooks. This is a no-op since hooks are active when registered."""
        pass
    
    def disable_hooks(self):
        """Disable all registered hooks. This is a no-op since hooks are managed per registration."""
        pass
    
    def get_captured_activations(self) -> dict:
        """Get all captured activations from registered hooks."""
        activations = {}
        for hook_id, hook in self.hooks.items():
            if hook.activations:
                activations[hook_id] = [act.to_dict() for act in hook.activations]
        return activations
    
    def clear_captured_activations(self):
        """Clear all captured activations from registered hooks."""
        for hook in self.hooks.values():
            hook.activations.clear()
    
    def remove_all_hooks(self):
        """Remove all registered hooks. Alias for clear_all_hooks."""
        self.clear_all_hooks()

    def unregister_hook(self, hook_id: str):
        """Unregister an activation hook and clean up any patched methods."""
        if hook_id in self._registered_handles:
            handle = self._registered_handles[hook_id]
            
            # Restore original __call__ method if we patched it
            if 'original_call' in handle and hasattr(handle['module'], '__call__'):
                handle['module'].__call__ = handle['original_call']
            
            # Remove any standard PyTorch-style hooks
            if 'handle' in handle and hasattr(handle['handle'], 'remove'):
                handle['handle'].remove()
            
            del self._registered_handles[hook_id]
        
        if hook_id in self.hooks:
            del self.hooks[hook_id]
    
    def clear_all_hooks(self):
        """Clear all registered hooks and clean up any patched methods."""
        for hook_id in list(self.hooks.keys()):
            self.unregister_hook(hook_id)
            
        # Clear any remaining hooks just to be safe
        self.hooks.clear()
        self._registered_handles.clear()
    
    def get_activations(self, hook_id: Optional[str] = None) -> Dict[str, List[CapturedActivation]]:
        """Get captured activations from hooks."""
        if hook_id:
            if hook_id in self.hooks:
                return {hook_id: self.hooks[hook_id].activations}
            else:
                return {}
        
        return {hid: hook.activations for hid, hook in self.hooks.items()}
    
    def clear_activations(self, hook_id: Optional[str] = None) -> None:
        """Clear captured activations for a specific hook or all hooks."""
        if hook_id:
            if hook_id in self.hooks:
                self.hooks[hook_id].clear()
        else:
            for hook in self.hooks.values():
                hook.clear()
    
    def _find_module(self, layer_name: str, component: Union[ComponentType, str, None] = None) -> Optional[Any]:
        """Find a module in the model by layer name and component type.
        
        This method handles complex component paths that may include list indices and nested attributes.
        If component is a string, it's treated as a path to traverse from the base layer.
        """
        print(f"\n[DEBUG] ====== _find_module ======")
        print(f"[DEBUG] layer_name: {layer_name}")
        print(f"[DEBUG] component: {component}")
        print(f"[DEBUG] Model type: {type(self.model).__name__}")
        
        # Start from the model
        current = self.model
        
        # Helper function to handle list indices in paths like 'layers[0]'
        def get_attr_or_item(obj, name):
            # Handle list indices like 'layers[0]'
            if '[' in name and name.endswith(']'):
                base_name = name.split('[')[0]
                idx = int(name.split('[')[1][:-1])
                if hasattr(obj, base_name):
                    obj = getattr(obj, base_name)
                    if isinstance(obj, (list, tuple)) and 0 <= idx < len(obj):
                        return obj[idx]
            # Handle regular attribute access
            if hasattr(obj, name):
                return getattr(obj, name)
            # Handle dictionary access
            if isinstance(obj, dict) and name in obj:
                return obj[name]
            # Handle list/tuple access by index
            if isinstance(obj, (list, tuple)) and name.isdigit() and 0 <= int(name) < len(obj):
                return obj[int(name)]
            return None
        
        try:
            # Traverse the layer path (e.g., 'model.layers')
            parts = [p for p in layer_name.split('.') if p]  # Remove empty parts
            for i, part in enumerate(parts):
                current = get_attr_or_item(current, part)
                print(f"[DEBUG]   Resolved {'.'.join(parts[:i+1])} -> {type(current).__name__}")
            
            # If we have a component, check if we need to resolve it further
            if component is not None:
                component_value = component.value if isinstance(component, ComponentType) else str(component)
                
                # Check if the layer_name already specifies the component (e.g., 'model.layers.0.self_attn')
                # In this case, we don't need to resolve the component further
                if any(comp in layer_name for comp in ['self_attn', 'mlp', 'attention', 'residual']):
                    print(f"[DEBUG] Layer name already includes component, using resolved module directly")
                else:
                    # Only resolve component if it's not already in the layer path
                    if hasattr(current, component_value):
                        current = getattr(current, component_value)
                        print(f"[DEBUG]   Resolved component {component_value} -> {type(current).__name__}")
                    else:
                        print(f"[DEBUG] Component {component_value} not found in {type(current).__name__}, using layer module directly")
            
            print(f"[DEBUG] Found module: {current}")
            print(f"[DEBUG] Module type: {type(current).__name__}")
            return current
            
        except (AttributeError, IndexError, KeyError, ValueError) as e:
            print(f"[ERROR] Failed to find module: {e}")
            print(f"[DEBUG] Current module: {current}")
            if hasattr(current, '__dict__'):
                print(f"[DEBUG] Module attributes: {[k for k in current.__dict__.keys() if not k.startswith('_')]}")
            elif isinstance(current, (list, tuple)):
                print(f"[DEBUG] List length: {len(current)}")
            elif isinstance(current, dict):
                print(f"[DEBUG] Dict keys: {list(current.keys())}")
            return None
        
        print(f"[DEBUG] Found module: {type(current).__name__}")
        print(f"[DEBUG] Module type: {type(current).__name__}")
        print(f"[DEBUG] Module has {len([attr for attr in dir(current) if not attr.startswith('_')])} attributes")
        
        # Check if the component exists in the module
        component_name = component.value if hasattr(component, 'value') else component
        if hasattr(current, component_name):
            print(f"[DEBUG] Found component '{component_name}' in module")
            current = getattr(current, component_name)
            print(f"[DEBUG] Component type: {type(current).__name__}")
            print(f"[DEBUG] Component attributes: {[attr for attr in dir(current) if not attr.startswith('_')]}")
        else:
            print(f"[WARNING] Component '{component_name}' not found in module, using module directly")
        
        return current
        
        try:
            # Handle direct layer access (e.g., 'layers.5' or 'transformer.h.5' for compatibility)
            if layer_name.startswith(('layers.', 'transformer.h.')):
                # Extract layer index (handling both 'layers.5' and 'transformer.h.5' formats)
                parts = layer_name.split('.')
                layer_idx = int(parts[1] if parts[0] == 'layers' else parts[2])
                
                # Get the layer from model.layers
                if not hasattr(self.model, 'layers'):
                    print(f"[ERROR] Model does not have 'layers' attribute. Available attributes: {[attr for attr in dir(self.model) if not attr.startswith('_')]}")
                    return None
                    
                if not (0 <= layer_idx < len(self.model.layers)):
                    print(f"[ERROR] Layer index {layer_idx} out of range for model.layers (length: {len(self.model.layers)})")
                    return None
                    
                print(f"[DEBUG] Accessing layer {layer_idx} from model.layers")
                    
                layer = self.model.layers[layer_idx]
                
                # Handle MoE architecture if present (common in MLX models)
                print(f"[DEBUG] Layer {layer_idx} attributes: {[attr for attr in dir(layer) if not attr.startswith('_')]}")
                
                if hasattr(layer, 'mlp'):
                    mlp = layer.mlp
                    print(f"[DEBUG] Found MLP in layer {layer_idx}")
                    print(f"[DEBUG] MLP attributes: {[attr for attr in dir(mlp) if not attr.startswith('_')]}")
                    
                    # Check for MLPBlock with experts (MoE)
                    if hasattr(mlp, 'experts') and hasattr(mlp, 'num_local_experts'):
                        if component == ComponentType.MOE:
                            return mlp
                        elif component == ComponentType.EXPERT:
                            # Handle expert access like 'layers.5.mlp.experts.0'
                            if len(parts) > 4 and parts[2] == 'mlp' and parts[3] == 'experts':
                                try:
                                    expert_idx = int(parts[4])
                                    if 0 <= expert_idx < mlp.num_local_experts:
                                        # Return a wrapper that can access the specific expert
                                        return ExpertWrapper(mlp.experts, expert_idx)
                                except (ValueError, IndexError) as e:
                                    print(f"Error accessing expert: {e}")
                            # Default to first expert if not specified or out of range
                            return ExpertWrapper(mlp.experts, 0) if mlp.num_local_experts > 0 else None
                    # Standard MLP (non-MoE)
                    elif component == ComponentType.MLP:
                        return mlp
                
                # Standard transformer components in MLX models
                if component == ComponentType.RESIDUAL:
                    return layer
                elif component == ComponentType.ATTENTION and hasattr(layer, 'self_attn'):
                    return layer.self_attn
                elif component == ComponentType.MLP and hasattr(layer, 'mlp'):
                    return layer.mlp
                elif component == ComponentType.LAYERNORM:
                    if hasattr(layer, 'input_layernorm'):
                        return layer.input_layernorm
                    elif hasattr(layer, 'ln_1'):
                        return layer.ln_1
                    elif hasattr(layer, 'pre_norm'):
                        return layer.pre_norm
                
                # For attention sub-components
                if component in [ComponentType.KEY, ComponentType.QUERY, ComponentType.VALUE] and hasattr(layer, 'self_attn'):
                    if hasattr(layer.self_attn, f'q_proj' if component == ComponentType.QUERY else 
                                            f'k_proj' if component == ComponentType.KEY else 'v_proj'):
                        return layer.self_attn
            
            # Handle embedding layer (common patterns in MLX models)
            print(f"[DEBUG] Checking for embedding layer: {layer_name}")
            if layer_name == 'embed_tokens':
                if hasattr(self.model, 'embed_tokens'):
                    print("[DEBUG] Found model.embed_tokens")
                    return self.model.embed_tokens
                elif hasattr(self.model, 'model') and hasattr(self.model.model, 'embed_tokens'):
                    print("[DEBUG] Found model.model.embed_tokens")
                    return self.model.model.embed_tokens
                else:
                    print("[DEBUG] No embed_tokens found in expected locations")
            
            # Handle direct attribute access as fallback
            parts = layer_name.split('.')
            module = self.model
            
            for part in parts:
                if not hasattr(module, part):
                    # Try to handle list/dict access
                    if isinstance(module, (list, tuple)) and part.isdigit():
                        return module[int(part)]
                    return None
                module = getattr(module, part)
            
            return module
        except (AttributeError, IndexError, KeyError, ValueError) as e:
            print(f"[ERROR] Failed to find module: {e}")
            return None


def serialize_activations(activations: Dict[str, List[Any]], format: str = 'numpy') -> Dict[str, Any]:
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