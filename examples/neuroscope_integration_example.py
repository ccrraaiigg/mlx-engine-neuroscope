"""
NeuroScope Integration Example

This example demonstrates how to use the extended MLX Engine with activation
capture capabilities for mechanistic interpretability analysis with NeuroScope.
"""

import sys
import json
from pathlib import Path

# Add the parent directory to the path so we can import mlx_engine
sys.path.insert(0, str(Path(__file__).parent.parent))

from mlx_engine import load_model, create_generator_with_activations, tokenize
from mlx_engine.activation_hooks_fixed import ComponentType
from neuroscope_integration import LMStudioNeuroScopeClient, NeuroScopeActivationBridge


def basic_activation_capture_example():
    """Basic example of capturing activations during generation."""
    print("=== Basic Activation Capture Example ===")
    
    # Note: Replace with actual model path
    model_path = "/path/to/your/model"  # e.g., "microsoft/DialoGPT-medium"
    
    try:
        # Load model
        print(f"Loading model from {model_path}...")
        model = load_model(model_path)
        print("Model loaded successfully!")
        
        # Define some activation hooks
        activation_hooks = [
            {
                'layer_name': 'transformer.h.5',
                'component': 'residual',
                'hook_id': 'layer_5_residual'
            },
            {
                'layer_name': 'transformer.h.10',
                'component': 'attention',
                'hook_id': 'layer_10_attention'
            }
        ]
        
        # Prepare input
        prompt = "The future of artificial intelligence will be"
        tokens = tokenize(model, prompt)
        print(f"Tokenized prompt: {len(tokens)} tokens")
        
        # Generate with activation capture
        print("Generating with activation capture...")
        full_text = ""
        all_activations = {}
        
        for result, activations in create_generator_with_activations(
            model, 
            tokens,
            activation_hooks=activation_hooks,
            max_tokens=50,
            temp=0.7
        ):
            full_text += result.text
            
            if activations:
                print(f"Captured activations from {len(activations)} hooks")
                for hook_id, hook_activations in activations.items():
                    if hook_id not in all_activations:
                        all_activations[hook_id] = []
                    all_activations[hook_id].extend(hook_activations)
            
            if result.stop_condition:
                break
        
        print(f"Generated text: {full_text}")
        print(f"Total captured activations: {len(all_activations)} hook types")
        
        # Analyze captured activations
        for hook_id, activations in all_activations.items():
            print(f"\nHook {hook_id}:")
            for i, activation in enumerate(activations):
                print(f"  Activation {i}: shape {activation['shape']}, dtype {activation['dtype']}")
        
    except Exception as e:
        print(f"Error in basic example: {e}")
        print("Note: Make sure to replace model_path with an actual model path")


def api_server_example():
    """Example using the REST API server."""
    print("\n=== API Server Example ===")
    
    # This example assumes the API server is running
    client = LMStudioNeuroScopeClient()
    
    if not client.health_check():
        print("API server is not running. Start it with:")
        print("python -m mlx_engine.api_server")
        return
    
    try:
        # List available models
        models = client.list_models()
        print(f"Available models: {len(models)}")
        
        # Example of loading a model (uncomment and modify as needed)
        # result = client.load_model("/path/to/your/model", model_id="test_model")
        # print(f"Model loading result: {result}")
        
        # Create activation hooks for analysis
        bridge = NeuroScopeActivationBridge()
        hooks = bridge.create_hook_specs_for_circuit_analysis(
            model_layers=12,  # Adjust based on your model
            components=['residual', 'attention']
        )
        
        print(f"Created {len(hooks)} hooks for circuit analysis")
        
        # Example messages for chat
        messages = [
            {"role": "user", "content": "Explain how attention mechanisms work in transformers."}
        ]
        
        # Generate with activations (uncomment when you have a loaded model)
        # result = client.generate_with_activations(
        #     messages=messages,
        #     activation_hooks=hooks[:4],  # Use first 4 hooks for demo
        #     max_tokens=100
        # )
        # 
        # print("Generation completed!")
        # print(f"Generated text: {result['choices'][0]['message']['content']}")
        # print(f"Captured activations from {len(result['activations'])} hooks")
        
    except Exception as e:
        print(f"Error in API example: {e}")


def neuroscope_bridge_example():
    """Example of converting activations for NeuroScope."""
    print("\n=== NeuroScope Bridge Example ===")
    
    # Create example activation data (simulating what MLX Engine would return)
    example_activations = {
        'layer_5_residual': [
            {
                'hook_id': 'layer_5_residual',
                'layer_name': 'transformer.h.5',
                'component': 'residual',
                'shape': [1, 10, 768],
                'dtype': 'float32',
                'is_input': False,
                'data': [[0.1, 0.2, 0.3] * 256] * 10  # Simplified example data
            }
        ],
        'layer_10_attention': [
            {
                'hook_id': 'layer_10_attention',
                'layer_name': 'transformer.h.10',
                'component': 'attention',
                'shape': [1, 10, 768],
                'dtype': 'float32',
                'is_input': False,
                'data': [[0.4, 0.5, 0.6] * 256] * 10  # Simplified example data
            }
        ]
    }
    
    # Convert for NeuroScope
    bridge = NeuroScopeActivationBridge()
    neuroscope_activations = bridge.convert_activations_for_neuroscope(example_activations)
    
    print("Converted activations for NeuroScope:")
    for hook_id, activations in neuroscope_activations.items():
        print(f"  {hook_id}: {len(activations)} activation tensors")
        for activation in activations:
            print(f"    Shape: {activation['shape']}, Component: {activation['component']}")


def generate_smalltalk_integration():
    """Generate Smalltalk code for NeuroScope integration."""
    print("\n=== Smalltalk Integration Code ===")
    
    from neuroscope_integration import generate_neuroscope_smalltalk_interface, LMStudioNeuroScopeClient
    
    client = LMStudioNeuroScopeClient()
    smalltalk_code = generate_neuroscope_smalltalk_interface(client)
    
    # Save to file
    output_file = Path(__file__).parent / "LMStudioRESTClient.st"
    with open(output_file, 'w') as f:
        f.write(smalltalk_code)
    
    print(f"Generated Smalltalk integration code saved to: {output_file}")
    print("\nTo use in NeuroScope:")
    print("1. Load the generated .st file into your Smalltalk environment")
    print("2. Create a client: client := LMStudioRESTClient default")
    print("3. Load a model: client loadModel: '/path/to/model'")
    print("4. Generate with activations using the provided methods")


def comprehensive_circuit_analysis_example():
    """Example of comprehensive circuit analysis setup."""
    print("\n=== Comprehensive Circuit Analysis Example ===")
    
    # Create hooks for different types of analysis
    bridge = NeuroScopeActivationBridge()
    
    # Circuit analysis hooks
    circuit_hooks = bridge.create_hook_specs_for_circuit_analysis(
        model_layers=24,
        components=['residual', 'attention', 'mlp']
    )
    
    # Attention analysis hooks
    attention_hooks = bridge.create_attention_analysis_hooks(
        model_layers=24,
        target_layers=[5, 10, 15, 20]  # Focus on specific layers
    )
    
    print(f"Circuit analysis hooks: {len(circuit_hooks)}")
    print(f"Attention analysis hooks: {len(attention_hooks)}")
    
    # Example of how you might organize different analysis types
    analysis_configs = {
        'basic_circuit_discovery': {
            'hooks': circuit_hooks[:10],  # First 10 for demo
            'description': 'Basic circuit discovery across key layers'
        },
        'attention_pattern_analysis': {
            'hooks': attention_hooks,
            'description': 'Detailed attention pattern analysis'
        },
        'residual_stream_analysis': {
            'hooks': [h for h in circuit_hooks if h['component'] == 'residual'],
            'description': 'Residual stream information flow analysis'
        }
    }
    
    for analysis_name, config in analysis_configs.items():
        print(f"\n{analysis_name}:")
        print(f"  Description: {config['description']}")
        print(f"  Hooks: {len(config['hooks'])}")
        print(f"  Example hook: {config['hooks'][0] if config['hooks'] else 'None'}")


if __name__ == '__main__':
    print("NeuroScope Integration Examples")
    print("=" * 50)
    
    # Run examples
    basic_activation_capture_example()
    api_server_example()
    neuroscope_bridge_example()
    generate_smalltalk_integration()
    comprehensive_circuit_analysis_example()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nNext steps:")
    print("1. Replace model paths with actual model locations")
    print("2. Start the API server: python -m mlx_engine.api_server")
    print("3. Load the generated Smalltalk code into NeuroScope")
    print("4. Begin your mechanistic interpretability analysis!")