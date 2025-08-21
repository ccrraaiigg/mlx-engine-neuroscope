from flask import request, jsonify, Response
from typing import Dict, List
import json
from mlx_engine import create_generator, create_generator_with_activations, tokenize
from .base import EndpointBase


class GenerationEndpoint(EndpointBase):
    """Handles text generation endpoints."""
    
    def __init__(self, api_instance):
        super().__init__(api_instance)
    
    def chat_completions(self):
        """Standard chat completions endpoint."""
        self.logger.info("=== CHAT COMPLETIONS REQUEST ===")
        self.logger.info(f"Request method: {request.method}")
        self.logger.info(f"Request headers: {dict(request.headers)}")
        self.logger.info(f"Content type: {request.content_type}")
        self.logger.info(f"Content length: {request.content_length}")
        
        data = request.get_json()
        self.logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        self.logger.info(f"Request data: {data}")
        
        if not data:
            self.logger.error("Missing request body")
            return jsonify(self.create_error_response(Exception('Request body is required'))), 400
        
        model_id = data.get('model', self.api.current_model)
        self.logger.info(f"Requested model: {model_id}")
        self.logger.info(f"Current model: {self.api.current_model}")
        self.logger.info(f"Available models: {list(self.api.models.keys())}")
        
        if not model_id or model_id not in self.api.models:
            self.logger.error(f"Model not found: {model_id}")
            return jsonify(self.create_error_response(Exception('Model not found'))), 404
        
        model = self.api.models[model_id]
        self.logger.info(f"Model type: {type(model).__name__}")
        self.logger.info(f"Model supports activations: {hasattr(model, 'activation_hook_manager')}")
        
        messages = data.get('messages', [])
        self.logger.info(f"Number of messages: {len(messages)}")
        self.logger.info(f"Messages: {messages}")
        
        if not messages:
            self.logger.error("No messages provided")
            return jsonify(self.create_error_response(Exception('Messages are required'))), 400
        
        # Convert messages to prompt
        self.logger.info("Converting messages to prompt...")
        prompt = self._messages_to_prompt(messages)
        self.logger.info(f"Generated prompt length: {len(prompt)}")
        self.logger.info(f"Generated prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Generated prompt: {prompt}")
        
        self.logger.info("Tokenizing prompt...")
        tokens = tokenize(model, prompt)
        self.logger.info(f"Token count: {len(tokens)}")
        
        # Generation parameters
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        stop = data.get('stop', [])
        stream = data.get('stream', False)
        
        generation_params = {
            'max_tokens': max_tokens,
            'temperature': temperature,
            'top_p': top_p,
            'stop': stop,
            'stream': stream
        }
        self.logger.info(f"Generation parameters: {generation_params}")
        
        try:
            self.logger.info("Starting generation process...")
            if stream:
                self.logger.info("Using streaming completion")
                response = Response(
                    self._stream_completion(model, tokens, max_tokens, temperature, top_p, stop),
                    mimetype='text/plain'
                )
                self.logger.info("Streaming response created successfully")
                self.logger.info("=== END CHAT COMPLETIONS (STREAMING) ===")
                return response
            else:
                self.logger.info("Using non-streaming completion")
                result = self._complete_generation(model, tokens, max_tokens, temperature, top_p, stop)
                self.logger.info("Generation completed successfully")
                self.logger.info("=== END CHAT COMPLETIONS (NON-STREAMING) ===")
                return result
                
        except Exception as e:
            import traceback
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'model_id': model_id,
                'current_model': self.api.current_model,
                'available_models': list(self.api.models.keys()),
                'prompt_length': len(prompt) if 'prompt' in locals() else 0,
                'traceback': traceback.format_exc(),
                'request_parameters': {
                    'max_tokens': data.get('max_tokens', 100),
                    'temperature': data.get('temperature', 0.7),
                    'top_p': data.get('top_p', 0.9),
                    'stop': data.get('stop')
                }
            }
            self.logger.error(f"Generation failed for model {model_id}: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify(self.create_error_response(Exception(error_details['error']))), 500
    
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
            self.logger.error(f"Error during generation with activations: {e}")
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


def register_generation_routes(app, api_instance):
    """Register generation routes."""
    endpoint = GenerationEndpoint(api_instance)
    
    app.add_url_rule('/v1/chat/completions', 'chat_completions', endpoint.chat_completions, methods=['POST'])