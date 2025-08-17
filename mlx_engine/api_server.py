"""
REST API Server for MLX Engine with NeuroScope Integration

This module provides a REST API server that extends the standard LM Studio
functionality to support activation capture for mechanistic interpretability
analysis with NeuroScope.
"""

from typing import Dict, List, Optional, Any, Iterator
import json
import asyncio
from pathlib import Path
import logging

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from flask import Flask, request, jsonify, Response, stream_template
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from mlx_engine import (
    load_model, 
    create_generator, 
    create_generator_with_activations,
    tokenize
)
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit
# Import activation hooks with debug logging
try:
    logger.info("Attempting to import from activation_hooks_fixed...")
    from mlx_engine.activation_hooks_fixed import serialize_activations, ActivationHookSpec
    logger.info(f"Successfully imported ActivationHookSpec: {ActivationHookSpec}")
    logger.info(f"ActivationHookSpec module: {ActivationHookSpec.__module__}")
    logger.info(f"ActivationHookSpec attributes: {dir(ActivationHookSpec)}")
except ImportError as e:
    logger.error(f"Failed to import from activation_hooks_fixed: {e}")
    raise

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MLXEngineAPI:
    """REST API server for MLX Engine with activation capture support."""
    
    def __init__(self):
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required for the API server. Install with: pip install flask flask-cors")
        
        self.app = Flask(__name__)
        CORS(self.app)  # Enable CORS for browser access
        
        self.models: Dict[str, ModelKit | VisionModelKit] = {}
        self.current_model: Optional[str] = None
        
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up API routes."""
        # Debug: Log all registered routes
        logger.info("Registering API routes...")
        
        @self.app.route('/health', methods=['GET'])
        def health():
            """Health check endpoint with version and timestamp."""
            import datetime
            logger.info(f"Health check endpoint called")
            return jsonify({
                'status': 'healthy', 
                'service': 'mlx-engine-neuroscope',
                'component': 'MLX Engine REST API',
                'version': '1.2.0',
                'timestamp': datetime.datetime.now().isoformat(),
                'current_model': self.current_model,
                'ready': self.current_model is not None
            })
            
        # Debug route to list all registered routes
        @self.app.route('/debug/routes', methods=['GET'])
        def debug_routes():
            """Debug endpoint to list all registered routes."""
            routes = []
            for rule in self.app.url_map.iter_rules():
                methods = ','.join(rule.methods)
                routes.append({
                    'endpoint': rule.endpoint,
                    'methods': methods,
                    'rule': str(rule)
                })
            return jsonify({'routes': routes})
        
        @self.app.route('/v1/models', methods=['GET'])
        def list_models():
            """List loaded models."""
            return jsonify({
                'models': [
                    {
                        'id': model_id,
                        'object': 'model',
                        'created': 0,  # Placeholder
                        'owned_by': 'mlx-engine'
                    }
                    for model_id in self.models.keys()
                ]
            })
        
        @self.app.route('/v1/models/load', methods=['POST'])
        def load_model_endpoint():
            """Load a model from path."""
            data = request.get_json()
            
            if not data or 'model_path' not in data:
                return jsonify({'error': 'model_path is required'}), 400
            
            model_path = data['model_path']
            model_id = data.get('model_id', Path(model_path).name)
            
            try:
                # Load model with optional parameters
                model = load_model(
                    model_path,
                    vocab_only=data.get('vocab_only', False),
                    max_kv_size=data.get('max_kv_size', 4096),
                    trust_remote_code=data.get('trust_remote_code', False),
                    kv_bits=data.get('kv_bits'),
                    kv_group_size=data.get('kv_group_size'),
                    quantized_kv_start=data.get('quantized_kv_start')
                )
                
                self.models[model_id] = model
                self.current_model = model_id
                
                return jsonify({
                    'model_id': model_id,
                    'status': 'loaded',
                    'supports_activations': hasattr(model, 'activation_hook_manager')
                })
                
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/v1/chat/completions', methods=['POST'])
        def chat_completions():
            """Standard chat completions endpoint."""
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'Request body is required'}), 400
            
            model_id = data.get('model', self.current_model)
            if not model_id or model_id not in self.models:
                return jsonify({'error': 'Model not found'}), 404
            
            model = self.models[model_id]
            messages = data.get('messages', [])
            
            if not messages:
                return jsonify({'error': 'Messages are required'}), 400
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            tokens = tokenize(model, prompt)
            
            # Generation parameters
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)
            stop = data.get('stop', [])
            stream = data.get('stream', False)
            
            try:
                if stream:
                    return Response(
                        self._stream_completion(model, tokens, max_tokens, temperature, top_p, stop),
                        mimetype='text/plain'
                    )
                else:
                    return self._complete_generation(model, tokens, max_tokens, temperature, top_p, stop)
                    
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/v1/activations/hooks', methods=['POST', 'DELETE'])
        def manage_activation_hooks():
            """Manage activation hooks for the model.
            
            POST: Register new activation hooks
            DELETE: Clear all activation hooks
            """
            # Get model from request or use default
            data = request.get_json() or {}
            model_id = data.get('model', self.current_model)
            
            # Debug logging
            logger.info(f"Activation hooks request - model_id: {model_id}")
            logger.info(f"Available models: {list(self.models.keys())}")
            logger.info(f"Current model: {self.current_model}")
            
            if not model_id or model_id not in self.models:
                return jsonify({'error': f'Model not found. Available models: {list(self.models.keys())}, requested: {model_id}'}), 404
                
            model = self.models[model_id]
            
            # Check if model supports activation capture
            if not hasattr(model, 'activation_hook_manager'):
                return jsonify({'error': 'Model does not support activation capture'}), 400
            
            if request.method == 'POST':
                # Register new activation hooks
                hooks = data.get('hooks', [])
                if not hooks:
                    return jsonify({'error': 'No hooks provided'}), 400
                
                try:
                    registered_hooks = []
                    for hook_spec in hooks:
                        # Convert dict to ActivationHookSpec if needed
                        if isinstance(hook_spec, dict):
                            hook_spec = ActivationHookSpec(
                                layer_name=hook_spec['layer_name'],
                                component=hook_spec.get('component'),
                                hook_id=hook_spec.get('hook_id'),
                                capture_input=hook_spec.get('capture_input', False),
                                capture_output=hook_spec.get('capture_output', True)
                            )
                        
                        # Register the hook
                        hook_id = model.activation_hook_manager.register_hook(hook_spec)
                        if hook_id:
                            registered_hooks.append(hook_id)
                    
                    return jsonify({
                        'status': 'hooks_registered',
                        'registered_hooks': registered_hooks
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to register activation hooks: {e}")
                    return jsonify({'error': str(e)}), 500
                    
            elif request.method == 'DELETE':
                # Clear all activation hooks
                try:
                    model.activation_hook_manager.clear_all_hooks()
                    return jsonify({'status': 'hooks_cleared'})
                except Exception as e:
                    logger.error(f"Failed to clear activation hooks: {e}")
                    return jsonify({'error': str(e)}), 500
        
        @self.app.route('/v1/chat/completions/with_activations', methods=['POST'])
        def chat_completions_with_activations():
            """Extended endpoint that captures activations during generation."""
            data = request.get_json()
            
            if not data:
                return jsonify({'error': 'Request body is required'}), 400
            
            model_id = data.get('model', self.current_model)
            if not model_id or model_id not in self.models:
                return jsonify({'error': 'Model not found'}), 404
            
            model = self.models[model_id]
            
            # Check if model supports activation capture
            if not hasattr(model, 'activation_hook_manager'):
                return jsonify({'error': 'Model does not support activation capture'}), 400
            
            messages = data.get('messages', [])
            activation_hooks = data.get('activation_hooks', [])
            
            if not messages:
                return jsonify({'error': 'Messages are required'}), 400
            
            if not activation_hooks:
                return jsonify({'error': 'activation_hooks are required for this endpoint'}), 400
            
            # Convert messages to prompt
            prompt = self._messages_to_prompt(messages)
            tokens = tokenize(model, prompt)
            
            # Generation parameters
            max_tokens = data.get('max_tokens', 100)
            temperature = data.get('temperature', 0.7)
            top_p = data.get('top_p', 0.9)
            stop = data.get('stop', [])
            stream = data.get('stream', False)
            
            try:
                # Clear GPU cache before generation to free memory
                import mlx.core as mx
                mx.clear_cache()
                
                if stream:
                    return Response(
                        self._stream_completion_with_activations(
                            model, tokens, activation_hooks, max_tokens, temperature, top_p, stop
                        ),
                        mimetype='application/x-ndjson'
                    )
                else:
                    # Use real activation capture now that memory issue is resolved
                    result = self._complete_generation_with_activations(
                        model, tokens, activation_hooks, max_tokens, temperature, top_p, stop
                    )
                    
                    # Clear cache after generation
                    mx.clear_cache()
                    return result
                    
            except Exception as e:
                logger.error(f"Generation with activations failed: {e}")
                # Clear cache on error too
                try:
                    import mlx.core as mx
                    mx.clear_cache()
                except:
                    pass
                return jsonify({'error': str(e)}), 500
        
        # Removed duplicate route registrations for /v1/activations/hooks
        # The functionality is now handled by the manage_activation_hooks endpoint
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to a single prompt string."""
        # Simple implementation - in practice, you'd want to handle different
        # chat templates based on the model
        prompt_parts = []
        for message in messages:
            role = message.get('role', 'user')
            content = message.get('content', '')
            if role == 'system':
                prompt_parts.append(f"System: {content}")
            elif role == 'user':
                prompt_parts.append(f"User: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"Assistant: {content}")
        
        return "\n".join(prompt_parts) + "\nAssistant:"
    
    def _complete_generation(self, model, tokens, max_tokens, temperature, top_p, stop):
        """Complete generation without streaming."""
        full_text = ""
        
        for result in create_generator(
            model, tokens,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            stop_strings=stop
        ):
            full_text += result.text
            
            if result.stop_condition:
                break
        
        return jsonify({
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': full_text
                },
                'finish_reason': 'stop'
            }],
            'usage': {
                'prompt_tokens': len(tokens),
                'completion_tokens': len(full_text.split()),  # Rough estimate
                'total_tokens': len(tokens) + len(full_text.split())
            }
        })
    
    def _stream_completion(self, model, tokens, max_tokens, temperature, top_p, stop):
        """Stream generation results."""
        for result in create_generator(
            model, tokens,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            stop_strings=stop
        ):
            chunk = {
                'choices': [{
                    'delta': {
                        'content': result.text
                    },
                    'finish_reason': None
                }]
            }
            
            yield f"data: {json.dumps(chunk)}\n\n"
            
            if result.stop_condition:
                final_chunk = {
                    'choices': [{
                        'delta': {},
                        'finish_reason': 'stop'
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                break
        
        yield "data: [DONE]\n\n"
    
    def _complete_generation_with_activations(self, model, tokens, activation_hooks, 
                                            max_tokens, temperature, top_p, stop):
        """Complete generation with activation capture."""
        full_text = ""
        all_activations = {}
        
        # Use the actual activation capture system
        try:
            for result, activations in create_generator_with_activations(
                model, tokens,
                activation_hooks=activation_hooks,
                max_tokens=max_tokens,
                temp=temperature,
                top_p=top_p,
                stop_strings=stop
            ):
                full_text += result.text
                
                if activations:
                    # Merge activations
                    for hook_id, hook_activations in activations.items():
                        if hook_id not in all_activations:
                            all_activations[hook_id] = []
                        all_activations[hook_id].extend(hook_activations)
                
                if result.stop_condition:
                    break
        
        except Exception as e:
            logger.error(f"Error during generation with activations: {e}")
            raise e
        
        return jsonify({
            'choices': [{
                'message': {
                    'role': 'assistant',
                    'content': full_text
                },
                'finish_reason': 'stop'
            }],
            'activations': all_activations,
            'usage': {
                'prompt_tokens': len(tokens),
                'completion_tokens': len(full_text.split()),
                'total_tokens': len(tokens) + len(full_text.split())
            }
        })
    
    def _stream_completion_with_activations(self, model, tokens, activation_hooks,
                                          max_tokens, temperature, top_p, stop):
        """Stream generation results with activations."""
        for result, activations in create_generator_with_activations(
            model, tokens,
            activation_hooks=activation_hooks,
            max_tokens=max_tokens,
            temp=temperature,
            top_p=top_p,
            stop_strings=stop
        ):
            chunk = {
                'choices': [{
                    'delta': {
                        'content': result.text
                    },
                    'finish_reason': None
                }],
                'activations': activations
            }
            
            yield f"{json.dumps(chunk)}\n"
            
            if result.stop_condition:
                final_chunk = {
                    'choices': [{
                        'delta': {},
                        'finish_reason': 'stop'
                    }],
                    'activations': None
                }
                yield f"{json.dumps(final_chunk)}\n"
                break
    
    def run(self, host='127.0.0.1', port=8080, debug=False):
        """Run the API server."""
        logger.info(f"Starting MLX Engine API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)


def create_app():
    """Factory function to create the Flask app."""
    api = MLXEngineAPI()
    return api.app


if __name__ == '__main__':
    api = MLXEngineAPI()
    api.run(debug=True)