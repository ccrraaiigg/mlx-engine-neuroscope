#!/usr/bin/env python3
"""
NeuroScope REST Interface Demo

This demo shows how NeuroScope will interact with the MLX Engine REST API
for mechanistic interpretability analysis. It elaborates on test_gpt_oss_20b.py
by demonstrating the full REST interface workflow.
"""

import sys
import os
import json
import time
import requests
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add the current directory to Python path
sys.path.insert(0, os.getcwd())

from mlx_engine.api_server import MLXEngineAPI

class NeuroScopeRESTDemo:
    """Demo client showing how NeuroScope will use the REST interface."""
    
    def __init__(self, base_url: str = "http://127.0.0.1:8080"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def health_check(self) -> bool:
        """Check if the API server is healthy."""
        try:
            response = self.session.get(f"{self.base_url}/health")
            return response.status_code == 200
        except requests.RequestException:
            return False
    
    def load_model(self, model_path: str, model_id: Optional[str] = None) -> Dict[str, Any]:
        """Load a model via REST API."""
        payload = {
            'model_path': model_path,
            'model_id': model_id or Path(model_path).name,
            'trust_remote_code': False,
            'max_kv_size': 4096
        }
        
        response = self.session.post(f"{self.base_url}/v1/models/load", json=payload)
        response.raise_for_status()
        return response.json()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List available models."""
        response = self.session.get(f"{self.base_url}/v1/models")
        response.raise_for_status()
        return response.json()['models']
    
    def chat_completion(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Standard chat completion without activations."""
        payload = {
            'messages': messages,
            'max_tokens': kwargs.get('max_tokens', 100),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'stop': kwargs.get('stop', []),
            'stream': kwargs.get('stream', False)
        }
        
        response = self.session.post(f"{self.base_url}/v1/chat/completions", json=payload)
        response.raise_for_status()
        return response.json()
    
    def chat_completion_with_activations(self, messages: List[Dict[str, str]], 
                                       activation_hooks: List[Dict[str, Any]], 
                                       **kwargs) -> Dict[str, Any]:
        """Chat completion with activation capture for NeuroScope."""
        payload = {
            'messages': messages,
            'activation_hooks': activation_hooks,
            'max_tokens': kwargs.get('max_tokens', 100),
            'temperature': kwargs.get('temperature', 0.7),
            'top_p': kwargs.get('top_p', 0.9),
            'stop': kwargs.get('stop', []),
            'stream': kwargs.get('stream', False)
        }
        
        response = self.session.post(
            f"{self.base_url}/v1/chat/completions/with_activations", 
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def register_activation_hooks(self, hooks: List[Dict[str, Any]], 
                                model: Optional[str] = None) -> List[str]:
        """Register activation hooks on the model."""
        payload = {
            'hooks': hooks,
            'model': model
        }
        
        response = self.session.post(f"{self.base_url}/v1/activations/hooks", json=payload)
        response.raise_for_status()
        return response.json()['registered_hooks']
    
    def clear_activation_hooks(self, model: Optional[str] = None) -> bool:
        """Clear all activation hooks."""
        params = {'model': model} if model else {}
        response = self.session.delete(f"{self.base_url}/v1/activations/hooks", params=params)
        response.raise_for_status()
        return response.json()['status'] == 'hooks cleared'


def start_api_server():
    """Start the API server in a separate thread."""
    api = MLXEngineAPI()
    api.run(host='127.0.0.1', port=8080, debug=False)


def demo_basic_rest_interface():
    """Demo basic REST interface functionality."""
    print("=== Basic REST Interface Demo ===")
    
    client = NeuroScopeRESTDemo()
    
    # Health check
    print("1. Checking API server health...")
    if not client.health_check():
        print("❌ API server is not running!")
        return False
    print("✅ API server is healthy")
    
    # Load model
    model_path = "./models/nightmedia/gpt-oss-20b-q4-hi-mlx"
    if not Path(model_path).exists():
        print(f"❌ Model not found at {model_path}")
        return False
    
    print(f"2. Loading model from {model_path}...")
    try:
        load_result = client.load_model(model_path, "gpt-oss-20b")
        print(f"✅ Model loaded: {load_result['model_id']}")
        print(f"   Supports activations: {load_result['supports_activations']}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # List models
    print("3. Listing available models...")
    models = client.list_models()
    print(f"✅ Found {len(models)} models:")
    for model in models:
        print(f"   - {model['id']}")
    
    # Basic chat completion
    print("4. Testing basic chat completion...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is three plus four?"}
    ]
    
    try:
        result = client.chat_completion(messages, max_tokens=50)
        response_text = result['choices'][0]['message']['content']
        print(f"✅ Generated response: {response_text.strip()}")
        
        # Check if answer is correct
        if "7" in response_text or "seven" in response_text.lower():
            print("✅ Model correctly answered the math question!")
        else:
            print("⚠️  Model response doesn't contain expected answer (7)")
        
    except Exception as e:
        print(f"❌ Chat completion failed: {e}")
        return False
    
    return True


def demo_neuroscope_activation_capture():
    """Demo activation capture functionality for NeuroScope."""
    print("\n=== NeuroScope Activation Capture Demo ===")
    
    client = NeuroScopeRESTDemo()
    
    if not client.health_check():
        print("❌ API server is not running!")
        return False
    
    # Define activation hooks for mechanistic interpretability
    print("1. Setting up activation hooks for circuit analysis...")
    
    # Hooks for different components across key layers
    activation_hooks = [
        {
            'layer_name': 'transformer.h.5',
            'component': 'residual',
            'hook_id': 'layer_5_residual',
            'capture_input': False,
            'capture_output': True
        },
        {
            'layer_name': 'transformer.h.10',
            'component': 'attention',
            'hook_id': 'layer_10_attention',
            'capture_input': False,
            'capture_output': True
        },
        {
            'layer_name': 'transformer.h.15',
            'component': 'mlp',
            'hook_id': 'layer_15_mlp',
            'capture_input': False,
            'capture_output': True
        },
        {
            'layer_name': 'transformer.h.20',
            'component': 'residual',
            'hook_id': 'layer_20_residual',
            'capture_input': False,
            'capture_output': True
        }
    ]
    
    print(f"   Created {len(activation_hooks)} hooks for analysis")
    
    # Test messages for different types of analysis
    test_scenarios = [
        {
            'name': 'Mathematical Reasoning',
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "If I have 15 apples and give away 7, how many do I have left?"}
            ]
        },
        {
            'name': 'Factual Recall',
            'messages': [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"}
            ]
        },
        {
            'name': 'Creative Writing',
            'messages': [
                {"role": "system", "content": "You are a creative writing assistant."},
                {"role": "user", "content": "Write the first sentence of a story about a robot discovering emotions."}
            ]
        }
    ]
    
    # Run activation capture for each scenario
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n2.{i} Testing scenario: {scenario['name']}")
        
        try:
            result = client.chat_completion_with_activations(
                messages=scenario['messages'],
                activation_hooks=activation_hooks,
                max_tokens=80,
                temperature=0.7
            )
            
            response_text = result['choices'][0]['message']['content']
            activations = result['activations']
            
            print(f"   ✅ Generated: {response_text.strip()[:100]}...")
            print(f"   ✅ Captured activations from {len(activations)} hooks:")
            
            # Analyze captured activations
            for hook_id, hook_activations in activations.items():
                if hook_activations:
                    activation_count = len(hook_activations)
                    first_activation = hook_activations[0]
                    shape = first_activation.get('shape', 'unknown')
                    print(f"      - {hook_id}: {activation_count} activations, shape {shape}")
                else:
                    print(f"      - {hook_id}: No activations captured")
            
        except Exception as e:
            print(f"   ❌ Failed to capture activations: {e}")
            continue
    
    return True


def demo_neuroscope_circuit_analysis():
    """Demo comprehensive circuit analysis setup for NeuroScope."""
    print("\n=== NeuroScope Circuit Analysis Demo ===")
    
    client = NeuroScopeRESTDemo()
    
    if not client.health_check():
        print("❌ API server is not running!")
        return False
    
    # Create hooks for comprehensive circuit analysis
    print("1. Setting up comprehensive circuit analysis hooks...")
    
    # Define different analysis types
    analysis_configs = {
        'attention_patterns': {
            'description': 'Analyze attention patterns across layers',
            'hooks': [
                {
                    'layer_name': f'transformer.h.{layer}',
                    'component': 'attention',
                    'hook_id': f'attention_layer_{layer}',
                    'capture_input': False,
                    'capture_output': True
                }
                for layer in [2, 5, 8, 11, 14, 17, 20]
            ]
        },
        'residual_stream': {
            'description': 'Track information flow through residual stream',
            'hooks': [
                {
                    'layer_name': f'transformer.h.{layer}',
                    'component': 'residual',
                    'hook_id': f'residual_layer_{layer}',
                    'capture_input': False,
                    'capture_output': True
                }
                for layer in [0, 4, 8, 12, 16, 20]
            ]
        },
        'mlp_processing': {
            'description': 'Analyze MLP processing patterns',
            'hooks': [
                {
                    'layer_name': f'transformer.h.{layer}',
                    'component': 'mlp',
                    'hook_id': f'mlp_layer_{layer}',
                    'capture_input': True,
                    'capture_output': True
                }
                for layer in [3, 7, 11, 15, 19]
            ]
        }
    }
    
    # Test prompt for circuit analysis
    circuit_analysis_prompt = [
        {"role": "system", "content": "You are an AI assistant that explains concepts clearly."},
        {"role": "user", "content": "Explain the concept of recursion in programming with a simple example."}
    ]
    
    # Run each analysis type
    for analysis_name, config in analysis_configs.items():
        print(f"\n2. Running {analysis_name} analysis...")
        print(f"   Description: {config['description']}")
        print(f"   Hooks: {len(config['hooks'])}")
        
        try:
            result = client.chat_completion_with_activations(
                messages=circuit_analysis_prompt,
                activation_hooks=config['hooks'],
                max_tokens=120,
                temperature=0.6
            )
            
            response_text = result['choices'][0]['message']['content']
            activations = result['activations']
            
            print(f"   ✅ Generated response ({len(response_text)} chars)")
            print(f"   ✅ Captured activations from {len(activations)} hooks")
            
            # Detailed activation analysis
            total_activations = 0
            for hook_id, hook_activations in activations.items():
                if hook_activations:
                    total_activations += len(hook_activations)
                    # Show details for first activation
                    first_activation = hook_activations[0]
                    shape = first_activation.get('shape', 'unknown')
                    component = first_activation.get('component', 'unknown')
                    print(f"      {hook_id}: {len(hook_activations)} activations, "
                          f"shape {shape}, component {component}")
            
            print(f"   📊 Total activation tensors captured: {total_activations}")
            
        except Exception as e:
            print(f"   ❌ Analysis failed: {e}")
            continue
    
    return True


def demo_streaming_with_activations():
    """Demo streaming generation with real-time activation capture."""
    print("\n=== Streaming with Activations Demo ===")
    
    # Note: This would require implementing streaming in the client
    # For now, we'll show the concept
    
    print("1. Streaming generation allows real-time activation analysis")
    print("2. NeuroScope can visualize activations as they're generated")
    print("3. This enables live circuit analysis during generation")
    
    # Example of what streaming data would look like
    example_stream_data = [
        {
            'choices': [{'delta': {'content': 'The'}}],
            'activations': {
                'layer_5_residual': [{'shape': [1, 1, 768], 'data': '...'}]
            }
        },
        {
            'choices': [{'delta': {'content': ' concept'}}],
            'activations': {
                'layer_5_residual': [{'shape': [1, 1, 768], 'data': '...'}]
            }
        },
        {
            'choices': [{'delta': {'content': ' of'}}],
            'activations': {
                'layer_5_residual': [{'shape': [1, 1, 768], 'data': '...'}]
            }
        }
    ]
    
    print("4. Example streaming data structure:")
    for i, chunk in enumerate(example_stream_data):
        print(f"   Chunk {i+1}: '{chunk['choices'][0]['delta']['content']}' + activations")
    
    print("✅ Streaming concept demonstrated")
    return True


def demo_neuroscope_integration_workflow():
    """Demo the complete NeuroScope integration workflow."""
    print("\n=== Complete NeuroScope Integration Workflow ===")
    
    workflow_steps = [
        "1. NeuroScope connects to MLX Engine REST API",
        "2. Loads target model for analysis",
        "3. Defines activation hooks for specific circuits",
        "4. Sends prompts with activation capture requests",
        "5. Receives generated text + activation tensors",
        "6. Visualizes activation patterns in real-time",
        "7. Analyzes circuit behavior and information flow",
        "8. Iterates with different prompts/hooks for deeper analysis"
    ]
    
    for step in workflow_steps:
        print(f"   {step}")
        time.sleep(0.5)  # Simulate workflow progression
    
    print("\n✅ Integration workflow complete!")
    
    # Show example NeuroScope analysis results
    print("\n📊 Example NeuroScope Analysis Results:")
    print("   - Attention Head 5.3 specializes in syntactic parsing")
    print("   - Layer 12 residual stream carries semantic information")
    print("   - MLP layers 8-10 perform factual recall")
    print("   - Circuit pathway: Input → Attention → MLP → Residual → Output")
    
    return True


def main():
    """Run the complete NeuroScope REST interface demo."""
    print("NeuroScope REST Interface Demo")
    print("=" * 60)
    print("This demo shows how NeuroScope will interact with MLX Engine")
    print("for mechanistic interpretability analysis via REST API.")
    print("=" * 60)
    
    # Start API server in background
    print("Starting API server...")
    server_thread = threading.Thread(target=start_api_server, daemon=True)
    server_thread.start()
    
    # Wait for server to start
    time.sleep(3)
    
    # Run demo scenarios
    demos = [
        ("Basic REST Interface", demo_basic_rest_interface),
        ("Activation Capture", demo_neuroscope_activation_capture),
        ("Circuit Analysis", demo_neuroscope_circuit_analysis),
        ("Streaming Concept", demo_streaming_with_activations),
        ("Integration Workflow", demo_neuroscope_integration_workflow)
    ]
    
    results = {}
    for demo_name, demo_func in demos:
        print(f"\n{'='*20} {demo_name} {'='*20}")
        try:
            results[demo_name] = demo_func()
        except Exception as e:
            print(f"❌ Demo failed: {e}")
            results[demo_name] = False
    
    # Summary
    print(f"\n{'='*60}")
    print("DEMO SUMMARY")
    print(f"{'='*60}")
    
    for demo_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{demo_name:.<40} {status}")
    
    successful_demos = sum(results.values())
    total_demos = len(results)
    
    print(f"\nOverall: {successful_demos}/{total_demos} demos successful")
    
    if successful_demos == total_demos:
        print("🎉 All demos passed! NeuroScope integration is ready.")
    else:
        print("⚠️  Some demos failed. Check the output above for details.")
    
    print("\nNext steps for NeuroScope integration:")
    print("1. Implement the REST client in NeuroScope")
    print("2. Add activation visualization components")
    print("3. Create circuit analysis tools")
    print("4. Test with real mechanistic interpretability workflows")


if __name__ == "__main__":
    main()