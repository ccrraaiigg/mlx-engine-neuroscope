"""
NeuroScope Integration for MLX Engine

This module provides integration between NeuroScope (Smalltalk-based mechanistic
interpretability framework) and the extended MLX Engine with activation capture
capabilities.
"""

import json
import requests
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import numpy as np
import base64


class LMStudioNeuroScopeClient:
    """Client for interfacing between NeuroScope and LM Studio with activation capture."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:50111"):
        """
        Initialize the client.
        
        Args:
            base_url: Base URL of the MLX Engine API server
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
    
    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def load_model(self, model_path: str, model_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Load a model for analysis.
        
        Args:
            model_path: Path to the model directory
            model_id: Optional identifier for the model
            **kwargs: Additional model loading parameters
        
        Returns:
            Dictionary with model loading results
        """
        payload = {
            'model_path': model_path,
            'model_id': model_id or Path(model_path).name,
            **kwargs
        }
        
        response = self.session.post(f"{self.base_url}/v1/models/load", json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()['models']
    
    def generate_with_activations(
        self,
        messages: List[Dict[str, str]],
        activation_hooks: List[Dict[str, Any]],
        model: Optional[str] = None,
        max_tokens: int = 100,
        temperature: float = 0.7,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generate text while capturing internal activations.
        
        Args:
            messages: Chat messages in OpenAI format
            activation_hooks: List of hook specifications
            model: Model identifier (optional)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            stream: Whether to stream results
        
        Returns:
            Dictionary containing generation results and captured activations
        """
        payload = {
            'messages': messages,
            'activation_hooks': activation_hooks,
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stream': stream
        }
        
        if model:
            payload['model'] = model
        
        if stop:
            payload['stop'] = stop
        
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions/with_activations",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def register_hooks(self, hooks: List[Dict[str, Any]], model: Optional[str] = None) -> List[str]:
        """
        Register activation hooks on a model.
        
        Args:
            hooks: List of hook specifications
            model: Model identifier (optional)
        
        Returns:
            List of registered hook IDs
        """
        payload = {'hooks': hooks}
        if model:
            payload['model'] = model
        
        response = self.session.post(f"{self.base_url}/v1/activations/hooks", json=payload)
        response.raise_for_status()
        return response.json()['registered_hooks']
    
    def clear_hooks(self, model: Optional[str] = None):
        """Clear all activation hooks from a model."""
        params = {}
        if model:
            params['model'] = model
        
        response = self.session.delete(f"{self.base_url}/v1/activations/hooks", params=params)
        response.raise_for_status()
        return response.json()


class NeuroScopeActivationBridge:
    """Bridge for converting between MLX Engine activations and NeuroScope format."""
    
    @staticmethod
    def convert_activations_for_neuroscope(activations: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert MLX Engine activation format to NeuroScope-compatible format.
        
        Args:
            activations: Raw activations from MLX Engine
        
        Returns:
            NeuroScope-compatible activation data
        """
        neuroscope_activations = {}
        
        for hook_id, activation_list in activations.items():
            neuroscope_activations[hook_id] = []
            
            for activation_data in activation_list:
                # Convert to NeuroScope ActivationTensor format
                neuroscope_tensor = {
                    'hook_id': activation_data['hook_id'],
                    'layer_name': activation_data['layer_name'],
                    'component': activation_data['component'],
                    'shape': activation_data['shape'],
                    'dtype': activation_data['dtype'],
                    'is_input': activation_data['is_input'],
                    'data': activation_data['data']  # Numpy array as list
                }
                
                neuroscope_activations[hook_id].append(neuroscope_tensor)
        
        return neuroscope_activations
    
    @staticmethod
    def create_hook_specs_for_circuit_analysis(
        model_layers: int,
        components: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Create hook specifications for comprehensive circuit analysis.
        
        Args:
            model_layers: Number of transformer layers in the model
            components: List of components to hook ('residual', 'attention', 'mlp')
        
        Returns:
            List of hook specifications
        """
        if components is None:
            components = ['residual', 'attention', 'mlp']
        
        hooks = []
        
        # Add embedding layer hook
        hooks.append({
            'layer_name': 'transformer.wte',
            'component': 'embedding',
            'hook_id': 'embedding_output'
        })
        
        # Add hooks for each transformer layer
        for layer_idx in range(model_layers):
            for component in components:
                hooks.append({
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': component,
                    'hook_id': f'layer_{layer_idx}_{component}',
                    'capture_output': True
                })
        
        return hooks
    
    @staticmethod
    def create_attention_analysis_hooks(
        model_layers: int,
        target_layers: Optional[List[int]] = None
    ) -> List[Dict[str, Any]]:
        """
        Create hook specifications for detailed attention analysis.
        
        Args:
            model_layers: Number of transformer layers
            target_layers: Specific layers to analyze (None for all)
        
        Returns:
            List of hook specifications for attention analysis
        """
        if target_layers is None:
            target_layers = list(range(model_layers))
        
        hooks = []
        
        for layer_idx in target_layers:
            # Hook attention scores and patterns
            hooks.extend([
                {
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': 'attention',
                    'hook_id': f'layer_{layer_idx}_attention_output'
                },
                {
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': 'attention_scores',
                    'hook_id': f'layer_{layer_idx}_attention_scores'
                },
                {
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': 'query',
                    'hook_id': f'layer_{layer_idx}_query'
                },
                {
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': 'key',
                    'hook_id': f'layer_{layer_idx}_key'
                },
                {
                    'layer_name': f'transformer.h.{layer_idx}',
                    'component': 'value',
                    'hook_id': f'layer_{layer_idx}_value'
                }
            ])
        
        return hooks


def generate_neuroscope_smalltalk_interface(client: LMStudioNeuroScopeClient) -> str:
    """
    Generate Smalltalk code for interfacing with the LM Studio client.
    
    This creates a Smalltalk class that NeuroScope can use to communicate
    with the extended LM Studio API.
    """
    
    smalltalk_code = '''
"LMStudioRESTClient - Interface between NeuroScope and LM Studio with activation capture"

Object subclass: #LMStudioRESTClient
    instanceVariableNames: 'baseUrl httpClient'
    classVariableNames: ''
    poolDictionaries: ''
    category: 'NeuroScope-Integration'

LMStudioRESTClient class >> default
    "Return a default client instance"
    ^self new baseUrl: 'http://127.0.0.1:50111'

LMStudioRESTClient >> baseUrl: aString
    "Set the base URL for the LM Studio API"
    baseUrl := aString.
    httpClient := HTTPClient new baseUrl: baseUrl

LMStudioRESTClient >> loadModel: modelPath
    "Load a model for analysis"
    | payload response |
    payload := Dictionary new
        at: #model_path put: modelPath;
        yourself.
    
    response := httpClient 
        post: '/v1/models/load'
        data: (JSON stringify: payload).
    
    ^JSON parse: response content

LMStudioRESTClient >> generateWithActivations: messages hooks: activationHooks
    "Generate text while capturing activations"
    ^self 
        generateWithActivations: messages 
        hooks: activationHooks 
        maxTokens: 100 
        temperature: 0.7

LMStudioRESTClient >> generateWithActivations: messages hooks: activationHooks maxTokens: maxTokens temperature: temperature
    "Generate text with full parameter control"
    | payload response |
    payload := Dictionary new
        at: #messages put: messages;
        at: #activation_hooks put: activationHooks;
        at: #max_tokens put: maxTokens;
        at: #temperature put: temperature;
        yourself.
    
    response := httpClient 
        post: '/v1/chat/completions/with_activations'
        data: (JSON stringify: payload).
    
    ^JSON parse: response content

LMStudioRESTClient >> createCircuitAnalysisHooks: numLayers
    "Create hook specifications for circuit analysis"
    | hooks |
    hooks := OrderedCollection new.
    
    "Add embedding hook"
    hooks add: (Dictionary new
        at: #layer_name put: 'transformer.wte';
        at: #component put: 'embedding';
        at: #hook_id put: 'embedding_output';
        yourself).
    
    "Add transformer layer hooks"
    0 to: numLayers - 1 do: [:layerIdx |
        #(residual attention mlp) do: [:component |
            hooks add: (Dictionary new
                at: #layer_name put: 'transformer.h.', layerIdx asString;
                at: #component put: component;
                at: #hook_id put: 'layer_', layerIdx asString, '_', component;
                yourself)]].
    
    ^hooks asArray

LMStudioRESTClient >> createAttentionAnalysisHooks: numLayers targetLayers: targetLayers
    "Create hook specifications for attention analysis"
    | hooks layers |
    hooks := OrderedCollection new.
    layers := targetLayers ifNil: [(0 to: numLayers - 1) asArray].
    
    layers do: [:layerIdx |
        #(attention attention_scores query key value) do: [:component |
            hooks add: (Dictionary new
                at: #layer_name put: 'transformer.h.', layerIdx asString;
                at: #component put: component;
                at: #hook_id put: 'layer_', layerIdx asString, '_', component;
                yourself)]].
    
    ^hooks asArray

LMStudioRESTClient >> convertActivationsForNeuroScope: activations
    "Convert MLX Engine activations to NeuroScope ActivationTensor objects"
    | neuroScopeActivations |
    neuroScopeActivations := Dictionary new.
    
    activations keysAndValuesDo: [:hookId :activationList |
        neuroScopeActivations at: hookId put: (
            activationList collect: [:activationData |
                ActivationTensor new
                    hookId: (activationData at: #hook_id);
                    layerName: (activationData at: #layer_name);
                    component: (activationData at: #component);
                    shape: (activationData at: #shape);
                    dtype: (activationData at: #dtype);
                    isInput: (activationData at: #is_input);
                    data: (WebGLTensor fromArray: (activationData at: #data));
                    yourself])].
    
    ^neuroScopeActivations
'''
    
    return smalltalk_code


# Example usage and testing
if __name__ == '__main__':
    # Example of how to use the integration
    client = LMStudioNeuroScopeClient()
    
    # Check if server is running
    if not client.health_check():
        print("MLX Engine API server is not running. Please start it first.")
        exit(1)
    
    # Example: Load a model (you would replace with actual model path)
    # result = client.load_model("/path/to/your/model")
    # print(f"Model loaded: {result}")
    
    # Example: Create hooks for circuit analysis
    bridge = NeuroScopeActivationBridge()
    hooks = bridge.create_hook_specs_for_circuit_analysis(
        model_layers=24,  # Adjust based on your model
        components=['residual', 'attention']
    )
    
    print(f"Created {len(hooks)} activation hooks for circuit analysis")
    
    # Example: Generate Smalltalk interface code
    smalltalk_code = generate_neuroscope_smalltalk_interface(client)
    print("Generated Smalltalk interface code:")
    print(smalltalk_code[:500] + "..." if len(smalltalk_code) > 500 else smalltalk_code)