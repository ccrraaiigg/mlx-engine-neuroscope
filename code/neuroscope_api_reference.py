#!/usr/bin/env python3
"""
NeuroScope API Reference

This script provides a comprehensive reference for the REST API endpoints
that NeuroScope will use for mechanistic interpretability analysis.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class ActivationHook:
    """Represents an activation hook configuration."""
    layer_name: str
    component: str  # 'attention', 'mlp', 'residual', 'embedding'
    hook_id: Optional[str] = None
    capture_input: bool = False
    capture_output: bool = True


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # 'system', 'user', 'assistant'
    content: str


@dataclass
class GenerationParams:
    """Generation parameters for text completion."""
    max_tokens: int = 100
    temperature: float = 0.7
    top_p: float = 0.9
    stop: List[str] = None
    stream: bool = False
    
    def __post_init__(self):
        if self.stop is None:
            self.stop = []


class NeuroScopeAPIReference:
    """Reference implementation showing NeuroScope API usage patterns."""
    
    @staticmethod
    def health_check_request() -> Dict[str, Any]:
        """GET /health - Check API server health."""
        return {
            'method': 'GET',
            'endpoint': '/health',
            'description': 'Check if the API server is running and healthy',
            'expected_response': {
                'status': 'healthy',
                'service': 'mlx-engine-neuroscope'
            }
        }
    
    @staticmethod
    def load_model_request(model_path: str, model_id: str = None) -> Dict[str, Any]:
        """POST /v1/models/load - Load a model for analysis."""
        return {
            'method': 'POST',
            'endpoint': '/v1/models/load',
            'description': 'Load a model from the specified path',
            'request_body': {
                'model_path': model_path,
                'model_id': model_id or 'default_model',
                'trust_remote_code': False,
                'max_kv_size': 4096,
                'vocab_only': False,
                'kv_bits': None,
                'kv_group_size': None,
                'quantized_kv_start': None
            },
            'expected_response': {
                'model_id': model_id or 'default_model',
                'status': 'loaded',
                'supports_activations': True
            }
        }
    
    @staticmethod
    def list_models_request() -> Dict[str, Any]:
        """GET /v1/models - List available models."""
        return {
            'method': 'GET',
            'endpoint': '/v1/models',
            'description': 'List all loaded models',
            'expected_response': {
                'models': [
                    {
                        'id': 'model_name',
                        'object': 'model',
                        'created': 0,
                        'owned_by': 'mlx-engine'
                    }
                ]
            }
        }
    
    @staticmethod
    def chat_completion_request(messages: List[ChatMessage], 
                              params: GenerationParams = None) -> Dict[str, Any]:
        """POST /v1/chat/completions - Standard chat completion."""
        if params is None:
            params = GenerationParams()
        
        return {
            'method': 'POST',
            'endpoint': '/v1/chat/completions',
            'description': 'Generate text completion without activation capture',
            'request_body': {
                'messages': [asdict(msg) for msg in messages],
                **asdict(params)
            },
            'expected_response': {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': 'Generated response text'
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'total_tokens': 30
                }
            }
        }
    
    @staticmethod
    def chat_completion_with_activations_request(
        messages: List[ChatMessage],
        activation_hooks: List[ActivationHook],
        params: GenerationParams = None
    ) -> Dict[str, Any]:
        """POST /v1/chat/completions/with_activations - Chat completion with activation capture."""
        if params is None:
            params = GenerationParams()
        
        return {
            'method': 'POST',
            'endpoint': '/v1/chat/completions/with_activations',
            'description': 'Generate text with activation capture for NeuroScope analysis',
            'request_body': {
                'messages': [asdict(msg) for msg in messages],
                'activation_hooks': [asdict(hook) for hook in activation_hooks],
                **asdict(params)
            },
            'expected_response': {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': 'Generated response text'
                    },
                    'finish_reason': 'stop'
                }],
                'activations': {
                    'hook_id_1': [
                        {
                            'hook_id': 'hook_id_1',
                            'layer_name': 'transformer.h.5',
                            'component': 'residual',
                            'shape': [1, 10, 768],
                            'dtype': 'float32',
                            'is_input': False,
                            'data': '...'  # Actual tensor data
                        }
                    ]
                },
                'usage': {
                    'prompt_tokens': 10,
                    'completion_tokens': 20,
                    'total_tokens': 30
                }
            }
        }
    
    @staticmethod
    def register_activation_hooks_request(hooks: List[ActivationHook]) -> Dict[str, Any]:
        """POST /v1/activations/hooks - Register activation hooks."""
        return {
            'method': 'POST',
            'endpoint': '/v1/activations/hooks',
            'description': 'Register activation hooks on the current model',
            'request_body': {
                'hooks': [asdict(hook) for hook in hooks],
                'model': 'optional_model_id'
            },
            'expected_response': {
                'registered_hooks': ['hook_id_1', 'hook_id_2', 'hook_id_3']
            }
        }
    
    @staticmethod
    def clear_activation_hooks_request() -> Dict[str, Any]:
        """DELETE /v1/activations/hooks - Clear activation hooks."""
        return {
            'method': 'DELETE',
            'endpoint': '/v1/activations/hooks',
            'description': 'Clear all activation hooks from the current model',
            'query_params': {
                'model': 'optional_model_id'
            },
            'expected_response': {
                'status': 'hooks cleared'
            }
        }


def generate_neuroscope_hook_configurations() -> Dict[str, List[ActivationHook]]:
    """Generate common activation hook configurations for NeuroScope analysis."""
    
    configurations = {}
    
    # Basic circuit discovery hooks
    configurations['circuit_discovery'] = [
        ActivationHook('transformer.h.0', 'residual', 'input_residual'),
        ActivationHook('transformer.h.5', 'attention', 'early_attention'),
        ActivationHook('transformer.h.5', 'mlp', 'early_mlp'),
        ActivationHook('transformer.h.10', 'attention', 'mid_attention'),
        ActivationHook('transformer.h.10', 'mlp', 'mid_mlp'),
        ActivationHook('transformer.h.15', 'attention', 'late_attention'),
        ActivationHook('transformer.h.15', 'mlp', 'late_mlp'),
        ActivationHook('transformer.h.20', 'residual', 'output_residual')
    ]
    
    # Attention pattern analysis
    configurations['attention_analysis'] = [
        ActivationHook(f'transformer.h.{i}', 'attention', f'attention_layer_{i}')
        for i in range(0, 24, 2)  # Every other layer
    ]
    
    # MLP processing analysis
    configurations['mlp_analysis'] = [
        ActivationHook(f'transformer.h.{i}', 'mlp', f'mlp_layer_{i}', 
                      capture_input=True, capture_output=True)
        for i in [3, 7, 11, 15, 19, 23]
    ]
    
    # Residual stream tracking
    configurations['residual_stream'] = [
        ActivationHook(f'transformer.h.{i}', 'residual', f'residual_layer_{i}')
        for i in range(0, 24, 4)  # Every 4th layer
    ]
    
    # Comprehensive analysis (all components, key layers)
    configurations['comprehensive'] = [
        ActivationHook('transformer.h.2', 'attention', 'comp_attention_2'),
        ActivationHook('transformer.h.2', 'mlp', 'comp_mlp_2'),
        ActivationHook('transformer.h.2', 'residual', 'comp_residual_2'),
        ActivationHook('transformer.h.8', 'attention', 'comp_attention_8'),
        ActivationHook('transformer.h.8', 'mlp', 'comp_mlp_8'),
        ActivationHook('transformer.h.8', 'residual', 'comp_residual_8'),
        ActivationHook('transformer.h.16', 'attention', 'comp_attention_16'),
        ActivationHook('transformer.h.16', 'mlp', 'comp_mlp_16'),
        ActivationHook('transformer.h.16', 'residual', 'comp_residual_16'),
        ActivationHook('transformer.h.22', 'attention', 'comp_attention_22'),
        ActivationHook('transformer.h.22', 'mlp', 'comp_mlp_22'),
        ActivationHook('transformer.h.22', 'residual', 'comp_residual_22')
    ]
    
    return configurations


def generate_neuroscope_test_scenarios() -> List[Dict[str, Any]]:
    """Generate test scenarios for NeuroScope analysis."""
    
    scenarios = [
        {
            'name': 'Mathematical Reasoning',
            'description': 'Test mathematical computation circuits',
            'messages': [
                ChatMessage('system', 'You are a helpful math tutor.'),
                ChatMessage('user', 'What is 127 + 348?')
            ],
            'expected_circuits': ['arithmetic_processing', 'numerical_encoding'],
            'analysis_focus': 'MLP layers for computation, attention for digit processing'
        },
        {
            'name': 'Factual Recall',
            'description': 'Test factual knowledge retrieval circuits',
            'messages': [
                ChatMessage('system', 'You are a knowledgeable assistant.'),
                ChatMessage('user', 'Who was the first person to walk on the moon?')
            ],
            'expected_circuits': ['factual_recall', 'entity_recognition'],
            'analysis_focus': 'Late layer MLPs for fact retrieval, attention for entity binding'
        },
        {
            'name': 'Language Understanding',
            'description': 'Test syntactic and semantic processing',
            'messages': [
                ChatMessage('system', 'You are a language expert.'),
                ChatMessage('user', 'Parse this sentence: "The quick brown fox jumps over the lazy dog."')
            ],
            'expected_circuits': ['syntactic_parsing', 'semantic_composition'],
            'analysis_focus': 'Early attention for syntax, mid-layer MLPs for semantics'
        },
        {
            'name': 'Logical Reasoning',
            'description': 'Test logical inference circuits',
            'messages': [
                ChatMessage('system', 'You are a logic expert.'),
                ChatMessage('user', 'If all birds can fly, and penguins are birds, can penguins fly?')
            ],
            'expected_circuits': ['logical_inference', 'contradiction_detection'],
            'analysis_focus': 'Late attention for logical connections, MLPs for inference'
        },
        {
            'name': 'Creative Generation',
            'description': 'Test creative and generative circuits',
            'messages': [
                ChatMessage('system', 'You are a creative writer.'),
                ChatMessage('user', 'Write a haiku about artificial intelligence.')
            ],
            'expected_circuits': ['creative_generation', 'pattern_completion'],
            'analysis_focus': 'Distributed processing across multiple layers'
        }
    ]
    
    return scenarios


def print_api_reference():
    """Print comprehensive API reference for NeuroScope integration."""
    
    print("NeuroScope MLX Engine API Reference")
    print("=" * 50)
    print(f"Script: {__file__}")
    print("=" * 50)
    
    api_ref = NeuroScopeAPIReference()
    
    # Health check
    health_req = api_ref.health_check_request()
    print(f"\n1. {health_req['description']}")
    print(f"   {health_req['method']} {health_req['endpoint']}")
    print(f"   Response: {json.dumps(health_req['expected_response'], indent=2)}")
    
    # Model loading
    load_req = api_ref.load_model_request("/path/to/model", "test_model")
    print(f"\n2. {load_req['description']}")
    print(f"   {load_req['method']} {load_req['endpoint']}")
    print(f"   Request: {json.dumps(load_req['request_body'], indent=2)}")
    print(f"   Response: {json.dumps(load_req['expected_response'], indent=2)}")
    
    # Chat completion with activations
    messages = [
        ChatMessage('system', 'You are helpful.'),
        ChatMessage('user', 'Hello!')
    ]
    hooks = [
        ActivationHook('transformer.h.5', 'attention', 'test_hook')
    ]
    
    chat_req = api_ref.chat_completion_with_activations_request(messages, hooks)
    print(f"\n3. {chat_req['description']}")
    print(f"   {chat_req['method']} {chat_req['endpoint']}")
    print(f"   Request: {json.dumps(chat_req['request_body'], indent=2)}")
    print("   Response: (truncated for brevity)")
    print(f"     - Generated text in choices[0].message.content")
    print(f"     - Activations in activations[hook_id]")
    
    # Hook configurations
    print(f"\n4. Common Hook Configurations")
    configs = generate_neuroscope_hook_configurations()
    for config_name, hook_list in configs.items():
        print(f"   {config_name}: {len(hook_list)} hooks")
        if hook_list:
            example_hook = asdict(hook_list[0])
            print(f"     Example: {json.dumps(example_hook, indent=6)}")
    
    # Test scenarios
    print(f"\n5. Test Scenarios for Analysis")
    scenarios = generate_neuroscope_test_scenarios()
    for scenario in scenarios:
        print(f"   {scenario['name']}: {scenario['description']}")
        print(f"     Focus: {scenario['analysis_focus']}")


def generate_neuroscope_client_template():
    """Generate a template for NeuroScope REST client implementation."""
    
    template = '''
class NeuroScopeMLXClient {
    constructor(baseUrl = "http://127.0.0.1:8080") {
        this.baseUrl = baseUrl;
    }
    
    async healthCheck() {
        const response = await fetch(`${this.baseUrl}/health`);
        return response.json();
    }
    
    async loadModel(modelPath, modelId = null) {
        const response = await fetch(`${this.baseUrl}/v1/models/load`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                model_path: modelPath,
                model_id: modelId,
                trust_remote_code: false,
                max_kv_size: 4096
            })
        });
        return response.json();
    }
    
    async generateWithActivations(messages, activationHooks, options = {}) {
        const response = await fetch(`${this.baseUrl}/v1/chat/completions/with_activations`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                messages: messages,
                activation_hooks: activationHooks,
                max_tokens: options.maxTokens || 100,
                temperature: options.temperature || 0.7,
                stream: options.stream || false
            })
        });
        return response.json();
    }
    
    // Helper methods for common analysis patterns
    createCircuitAnalysisHooks(layers = [5, 10, 15, 20]) {
        return layers.flatMap(layer => [
            {
                layer_name: `transformer.h.${layer}`,
                component: 'attention',
                hook_id: `attention_${layer}`,
                capture_output: true
            },
            {
                layer_name: `transformer.h.${layer}`,
                component: 'mlp',
                hook_id: `mlp_${layer}`,
                capture_output: true
            },
            {
                layer_name: `transformer.h.${layer}`,
                component: 'residual',
                hook_id: `residual_${layer}`,
                capture_output: true
            }
        ]);
    }
}
'''
    
    return template


def main():
    """Main function to display API reference and generate templates."""
    
    print_api_reference()
    
    print(f"\n{'='*50}")
    print("JavaScript Client Template")
    print(f"{'='*50}")
    
    template = generate_neuroscope_client_template()
    print(template)
    
    print(f"\n{'='*50}")
    print("Integration Notes")
    print(f"{'='*50}")
    
    notes = [
        "1. All activation data includes shape, dtype, and tensor values",
        "2. Hook IDs should be unique across all registered hooks",
        "3. Layer names follow the model's internal naming convention",
        "4. Components: 'attention', 'mlp', 'residual', 'embedding'",
        "5. Streaming mode provides real-time activation capture",
        "6. Large activation tensors may be compressed or chunked",
        "7. Error handling should account for model loading failures",
        "8. Memory management is important for large-scale analysis"
    ]
    
    for note in notes:
        print(f"   {note}")
    
    print(f"\n✅ API reference complete!")
    print("Use this reference to implement NeuroScope integration.")


if __name__ == "__main__":
    main()