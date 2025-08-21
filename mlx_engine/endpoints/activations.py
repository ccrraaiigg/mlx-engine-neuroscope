"""Activation analysis endpoints for the MLX Engine API.

This module handles activation capture, analysis, and patching endpoints.
"""

import logging
import time
import traceback
from typing import Dict, List, Optional, Any

from flask import request, jsonify, Response

from .base import EndpointBase

# Import activation-related components
try:
    from mlx_engine.activation_hooks import serialize_activations, ActivationHookSpec
except ImportError:
    ActivationHookSpec = None

try:
    from mlx_engine.activation_patching import (
        ActivationPatcher, CausalTracer, GradientBasedAttribution,
        InterventionType, InterventionSpec, CausalTracingResult,
        ComponentType, create_sophisticated_circuit_discovery_pipeline
    )
except ImportError:
    ActivationPatcher = None
    InterventionType = None
    InterventionSpec = None

try:
    from mlx_engine.attention_analysis import (
        AttentionAnalyzer, AttentionAnalysisResult, AttentionPatternType,
        AttentionScope, AttentionHead, AttentionPattern, CrossLayerDependency,
        create_attention_analysis_pipeline
    )
    ATTENTION_ANALYSIS_AVAILABLE = True
except ImportError:
    ATTENTION_ANALYSIS_AVAILABLE = False

from mlx_engine import tokenize

logger = logging.getLogger(__name__)


class ActivationEndpoint(EndpointBase):
    """Handles activation capture, analysis, and patching endpoints."""
    
    def __init__(self, api_instance):
        super().__init__(api_instance)
    
    def manage_activation_hooks(self):
        """Manage activation hooks for the model.
        
        POST: Register new activation hooks
        DELETE: Clear all activation hooks
        """
        logger.info(f"=== ACTIVATION HOOKS REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Content length: {request.content_length}")
        
        # Get model from request or use default
        data = request.get_json() or {}
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        logger.info(f"Request data: {data}")
        
        model_id = data.get('model', self.api.current_model)
        logger.info(f"Requested model: {model_id}")
        logger.info(f"Available models: {list(self.api.models.keys())}")
        logger.info(f"Current model: {self.api.current_model}")
        
        if not model_id or model_id not in self.api.models:
            logger.error(f"Model not found: {model_id}")
            return jsonify({'error': f'Model not found. Available models: {list(self.api.models.keys())}, requested: {model_id}'}), 404
            
        model = self.api.models[model_id]
        logger.info(f"Model type: {type(model).__name__}")
        logger.info(f"Model supports activations: {hasattr(model, 'activation_hook_manager')}")
        
        # Check if model supports activation capture
        if not hasattr(model, 'activation_hook_manager'):
            logger.error(f"Model does not support activation capture: {model_id}")
            return jsonify({'error': 'Model does not support activation capture'}), 400
        
        if request.method == 'POST':
            logger.info("Processing POST request - registering activation hooks")
            # Register new activation hooks
            hooks = data.get('hooks', [])
            logger.info(f"Number of hooks to register: {len(hooks)}")
            logger.info(f"Hook specifications: {hooks}")
            
            if not hooks:
                logger.error("No hooks provided in request")
                return jsonify({'error': 'No hooks provided'}), 400
            
            try:
                logger.info("Starting hook registration process...")
                registered_hooks = []
                for i, hook_spec in enumerate(hooks):
                    logger.info(f"Processing hook {i+1}/{len(hooks)}: {hook_spec}")
                    
                    # Convert dict to ActivationHookSpec if needed
                    if isinstance(hook_spec, dict):
                        logger.info(f"Converting dict to ActivationHookSpec for hook {i+1}")
                        hook_spec = ActivationHookSpec(
                            layer_name=hook_spec['layer_name'],
                            component=hook_spec.get('component'),
                            hook_id=hook_spec.get('hook_id'),
                            capture_input=hook_spec.get('capture_input', False),
                            capture_output=hook_spec.get('capture_output', True)
                        )
                        logger.info(f"Created ActivationHookSpec: {hook_spec}")
                    
                    # Register the hook
                    logger.info(f"Registering hook {i+1} with activation_hook_manager...")
                    hook_id = model.activation_hook_manager.register_hook(hook_spec)
                    logger.info(f"Hook {i+1} registration result: {hook_id}")
                    
                    if hook_id:
                        registered_hooks.append(hook_id)
                        logger.info(f"Successfully registered hook {i+1} with ID: {hook_id}")
                    else:
                        logger.warning(f"Failed to register hook {i+1}")
                
                response_data = {
                    'status': 'hooks_registered',
                    'registered_hooks': registered_hooks
                }
                logger.info(f"Hook registration completed. Response: {response_data}")
                logger.info(f"=== END ACTIVATION HOOKS (POST) ===")
                return jsonify(response_data)
                
            except Exception as e:
                error_details = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'model_id': model_id,
                    'hooks_attempted': len(hooks),
                    'traceback': traceback.format_exc()
                }
                logger.error(f"Failed to register activation hooks: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"=== END ACTIVATION HOOKS (POST ERROR) ===")
                return jsonify(error_details), 500
                
        elif request.method == 'DELETE':
            logger.info("Processing DELETE request - clearing activation hooks")
            # Clear all activation hooks
            try:
                logger.info("Calling clear_all_hooks on activation_hook_manager...")
                model.activation_hook_manager.clear_all_hooks()
                logger.info("Successfully cleared all activation hooks")
                
                response_data = {'status': 'hooks_cleared'}
                logger.info(f"Clear hooks response: {response_data}")
                logger.info(f"=== END ACTIVATION HOOKS (DELETE) ===")
                return jsonify(response_data)
            except Exception as e:
                error_details = {
                    'error': str(e),
                    'error_type': type(e).__name__,
                    'model_id': model_id,
                    'traceback': traceback.format_exc()
                }
                logger.error(f"Failed to clear activation hooks: {e}")
                logger.error(f"Full traceback: {traceback.format_exc()}")
                logger.error(f"=== END ACTIVATION HOOKS (DELETE ERROR) ===")
                return jsonify(error_details), 500
    
    def analyze_residual(self):
        """Analyze residual stream flow data.
        
        Expects POST data with:
        - residual_data: The residual stream data from activation capture
        - analysis_type: Type of analysis (e.g., 'residual_flow')
        """
        logger.info(f"=== ANALYZE RESIDUAL REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Content length: {request.content_length}")
        
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        logger.info(f"Request data (truncated): {str(data)[:500]}..." if data and len(str(data)) > 500 else f"Request data: {data}")
        
        if not data:
            logger.error("Missing request body")
            return jsonify({'error': 'Request body is required'}), 400
        
        residual_data = data.get('residual_data')
        analysis_type = data.get('analysis_type', 'residual_flow')
        
        logger.info(f"Analysis type: {analysis_type}")
        logger.info(f"Residual data present: {residual_data is not None}")
        if residual_data:
            logger.info(f"Residual data type: {type(residual_data)}")
            logger.info(f"Residual data size: {len(residual_data) if hasattr(residual_data, '__len__') else 'unknown'}")
        
        if not residual_data:
            logger.error("Missing residual_data in request")
            return jsonify({'error': 'residual_data is required'}), 400
        
        try:
            start_time = time.time()
            logger.info(f"Starting residual stream analysis at {start_time}")
            
            # Analyze the residual stream data
            logger.info("Calling _analyze_residual_stream...")
            analysis_result = self.api._analyze_residual_stream(residual_data, analysis_type)
            logger.info(f"Analysis completed. Result type: {type(analysis_result)}")
            logger.info(f"Analysis result keys: {list(analysis_result.keys()) if isinstance(analysis_result, dict) else 'Not a dict'}")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Analysis execution time: {execution_time_ms}ms")
            
            response_data = {
                'success': True,
                'analysis_type': analysis_type,
                'flow_data': analysis_result.get('flow_data', {}),
                'layer_contributions': analysis_result.get('layer_contributions', {}),
                'information_flow': analysis_result.get('information_flow', {}),
                'execution_time_ms': execution_time_ms
            }
            
            logger.info(f"Response data keys: {list(response_data.keys())}")
            logger.info(f"Flow data size: {len(response_data['flow_data']) if response_data['flow_data'] else 0}")
            logger.info(f"Layer contributions size: {len(response_data['layer_contributions']) if response_data['layer_contributions'] else 0}")
            logger.info(f"Information flow size: {len(response_data['information_flow']) if response_data['information_flow'] else 0}")
            logger.info(f"=== END ANALYZE RESIDUAL ===")
            
            return jsonify(response_data)
            
        except Exception as e:
            error_details = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'analysis_type': analysis_type,
                'traceback': traceback.format_exc()
            }
            logger.error(f"Failed to analyze residual stream: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"=== END ANALYZE RESIDUAL (ERROR) ===")
            return jsonify(error_details), 500
    
    def analyze_attention(self):
        """Analyze attention patterns in the model.
        
        Expects POST data with:
        - prompt: Text to analyze
        - layers: Optional list of layer indices to analyze
        - scope: Analysis scope (head_level, layer_level, cross_layer, global)
        """
        logger.info(f"=== ANALYZE ATTENTION REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Content length: {request.content_length}")
        
        if not ATTENTION_ANALYSIS_AVAILABLE:
            logger.error("Attention analysis not available")
            return jsonify({'error': 'Attention analysis not available'}), 503
        
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        logger.info(f"Request data (truncated): {str(data)[:500]}..." if data and len(str(data)) > 500 else f"Request data: {data}")
        
        if not data:
            logger.error("Missing request body")
            return jsonify({'error': 'Request body is required'}), 400
        
        model_id = data.get('model', self.api.current_model)
        logger.info(f"Requested model: {model_id}")
        logger.info(f"Current model: {self.api.current_model}")
        logger.info(f"Available models: {list(self.api.models.keys())}")
        
        if not model_id or model_id not in self.api.models:
            logger.error(f"Model not found: {model_id}")
            return jsonify({'error': 'Model not found'}), 404
        
        model = self.api.models[model_id]
        prompt = data.get('prompt')
        
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Prompt: {prompt}")
        logger.info(f"Prompt length: {len(prompt) if prompt else 0}")
        
        if not prompt:
            logger.error("Missing prompt in request")
            return jsonify({'error': 'prompt is required'}), 400
        
        # Parse parameters
        layers = data.get('layers')
        scope_str = data.get('scope', 'head_level')
        
        logger.info(f"Layers parameter: {layers}")
        logger.info(f"Scope parameter: {scope_str}")
        
        try:
            scope = AttentionScope(scope_str)
            logger.info(f"Parsed scope: {scope}")
        except ValueError as e:
            logger.error(f"Invalid scope: {scope_str}, error: {e}")
            return jsonify({'error': f'Invalid scope: {scope_str}. Must be one of: head_level, layer_level, cross_layer, global'}), 400
        
        try:
            start_time = time.time()
            logger.info(f"Starting attention analysis at {start_time}")
            
            # Create attention analyzer
            logger.info("Creating attention analysis pipeline...")
            analyzer = create_attention_analysis_pipeline(model)
            logger.info(f"Analyzer created: {type(analyzer)}")
            
            # Perform attention analysis
            logger.info("Starting attention pattern analysis...")
            result = analyzer.analyze_attention_patterns(
                prompt=prompt,
                layers=layers,
                scope=scope
            )
            logger.info(f"Analysis completed. Result type: {type(result)}")
            
            execution_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Attention analysis execution time: {execution_time_ms}ms")
            
            # Convert result to JSON-serializable format
            logger.info("Converting analysis result to JSON-serializable format...")
            logger.info(f"Result metadata: {result.metadata}")
            logger.info(f"Number of attention heads: {len(result.attention_heads)}")
            logger.info(f"Number of attention patterns: {len(result.attention_patterns)}")
            logger.info(f"Number of cross-layer dependencies: {len(result.cross_layer_dependencies)}")
            
            response_data = {
                'success': True,
                'prompt': prompt,
                'scope': scope.value,
                'analyzed_layers': result.metadata.get('analyzed_layers', []),
                'attention_heads': [
                    {
                        'layer': head.layer,
                        'head': head.head,
                        'pattern_type': head.pattern_type.value,
                        'confidence': head.confidence,
                        'circuit_role': head.circuit_role,
                        'key_tokens': head.key_tokens,
                        'value_tokens': head.value_tokens,
                        'metadata': head.metadata
                    }
                    for head in result.attention_heads
                ],
                'attention_patterns': [
                    {
                        'pattern_type': pattern.pattern_type.value,
                        'strength': pattern.strength,
                        'consistency': pattern.consistency,
                        'head_count': len(pattern.heads),
                        'description': pattern.description,
                        'token_positions': pattern.token_positions
                    }
                    for pattern in result.attention_patterns
                ],
                'cross_layer_dependencies': [
                    {
                        'source_layer': dep.source_layer,
                        'target_layer': dep.target_layer,
                        'dependency_type': dep.dependency_type,
                        'strength': dep.strength,
                        'mechanism': dep.mechanism,
                        'affected_heads': dep.affected_heads
                    }
                    for dep in result.cross_layer_dependencies
                ],
                'token_attributions': result.token_attributions,
                'circuit_components': result.circuit_components,
                'pattern_statistics': result.pattern_statistics,
                'visualization_data': result.visualization_data,
                'execution_time_ms': execution_time_ms,
                'metadata': result.metadata
            }
            
            logger.info(f"Response data keys: {list(response_data.keys())}")
            logger.info(f"Response data size estimate: {len(str(response_data))} characters")
            logger.info(f"=== END ANALYZE ATTENTION ===")
            
            return jsonify(response_data)
            
        except Exception as e:
            error_details = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'prompt': prompt,
                'scope': scope_str,
                'traceback': traceback.format_exc()
            }
            logger.error(f"Failed to analyze attention patterns: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error(f"=== END ANALYZE ATTENTION (ERROR) ===")
            return jsonify(error_details), 500
    
    def chat_completions_with_activations(self):
        """Extended endpoint that captures activations during generation."""
        logger.info(f"=== CHAT COMPLETIONS WITH ACTIVATIONS REQUEST ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content type: {request.content_type}")
        logger.info(f"Content length: {request.content_length}")
        
        data = request.get_json()
        logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
        logger.info(f"Request data (truncated): {str(data)[:500]}..." if data and len(str(data)) > 500 else f"Request data: {data}")
        
        if not data:
            logger.error("Missing request body")
            return jsonify({'error': 'Request body is required'}), 400
        
        model_id = data.get('model', self.api.current_model)
        logger.info(f"Requested model: {model_id}")
        logger.info(f"Current model: {self.api.current_model}")
        logger.info(f"Available models: {list(self.api.models.keys())}")
        
        if not model_id or model_id not in self.api.models:
            logger.error(f"Model not found: {model_id}")
            return jsonify({'error': 'Model not found'}), 404
        
        model = self.api.models[model_id]
        logger.info(f"Model type: {type(model)}")
        logger.info(f"Model supports activation capture: {hasattr(model, 'activation_hook_manager')}")
        
        # Check if model supports activation capture
        logger.info("Checking model activation capture support...")
        if not hasattr(model, 'activation_hook_manager'):
            logger.error("Model does not support activation capture")
            return jsonify({'error': 'Model does not support activation capture'}), 400
        
        messages = data.get('messages', [])
        activation_hooks = data.get('activation_hooks', [])
        
        logger.info(f"Number of messages: {len(messages)}")
        logger.info(f"Number of activation hooks: {len(activation_hooks)}")
        logger.info(f"Messages content: {[msg.get('content', '')[:100] + '...' if len(msg.get('content', '')) > 100 else msg.get('content', '') for msg in messages]}")
        logger.info(f"Activation hooks: {activation_hooks}")
        
        if not messages:
            logger.error("Missing messages in request")
            return jsonify({'error': 'Messages are required'}), 400
        
        if not activation_hooks:
            logger.error("Missing activation_hooks in request")
            return jsonify({'error': 'activation_hooks are required for this endpoint'}), 400
        
        # Convert messages to prompt
        logger.info("Converting messages to prompt...")
        prompt = self.api._messages_to_prompt(messages)
        logger.info(f"Generated prompt: {prompt[:200]}..." if len(prompt) > 200 else f"Generated prompt: {prompt}")
        
        logger.info("Tokenizing prompt...")
        tokens = tokenize(model, prompt)
        logger.info(f"Token count: {len(tokens)}")
        
        # Generation parameters
        max_tokens = data.get('max_tokens', 100)
        temperature = data.get('temperature', 0.7)
        top_p = data.get('top_p', 0.9)
        stop = data.get('stop', [])
        stream = data.get('stream', False)
        
        logger.info(f"Generation parameters - max_tokens: {max_tokens}, temperature: {temperature}, top_p: {top_p}, stop: {stop}, stream: {stream}")
        
        try:
            # Clear GPU cache before generation to free memory
            logger.info("Clearing GPU cache before generation...")
            import mlx.core as mx
            mx.clear_cache()
            logger.info("GPU cache cleared")
            
            if stream:
                logger.info("Starting streaming completion with activations...")
                logger.info(f"=== END CHAT COMPLETIONS WITH ACTIVATIONS (STREAMING) ===")
                return Response(
                    self.api._stream_completion_with_activations(
                        model, tokens, activation_hooks, max_tokens, temperature, top_p, stop
                    ),
                    mimetype='application/x-ndjson'
                )
            else:
                logger.info("Starting non-streaming completion with activations...")
                # Use real activation capture now that memory issue is resolved
                result = self.api._complete_generation_with_activations(
                    model, tokens, activation_hooks, max_tokens, temperature, top_p, stop
                )
                logger.info("Generation with activations completed")
                
                # Clear cache after generation
                logger.info("Clearing GPU cache after generation...")
                mx.clear_cache()
                logger.info("GPU cache cleared")
                logger.info(f"=== END CHAT COMPLETIONS WITH ACTIVATIONS ===")
                return result
                
        except Exception as e:
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'model_id': model_id,
                'prompt_length': len(prompt),
                'token_count': len(tokens),
                'activation_hooks_count': len(activation_hooks),
                'traceback': traceback.format_exc()
            }
            logger.error(f"Failed to complete generation with activations: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # Clear cache on error too
            try:
                import mlx.core as mx
                mx.clear_cache()
            except:
                pass
            
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'model_id': model_id,
                'current_model': self.api.current_model,
                'available_models': list(self.api.models.keys()),
                'prompt_length': len(prompt) if 'prompt' in locals() else 0,
                'activation_hooks_count': len(activation_hooks) if activation_hooks else 0,
                'traceback': traceback.format_exc(),
                'request_parameters': {
                    'max_tokens': data.get('max_tokens', 100),
                    'temperature': data.get('temperature', 0.7),
                    'top_p': data.get('top_p', 0.9),
                    'stop': data.get('stop'),
                    'activation_hooks': activation_hooks
                }
            }
            logger.error(f"Generation with activations failed for model {model_id}: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify(error_details), 500
    
    def activation_patch(self):
        """Apply activation patching for causal intervention analysis."""
        logger.info(f"=== ACTIVATION_PATCH START - Method: {request.method} ===")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Content-Type: {request.content_type}")
        logger.info(f"Content-Length: {request.content_length}")
        
        if not ActivationPatcher:
            logger.error("Activation patching not available - ActivationPatcher is None")
            return jsonify({'error': 'Activation patching not available'}), 501
            
        data = request.get_json()
        if not data:
            logger.error("Missing request body")
            return jsonify({'error': 'Request body is required'}), 400
            
        logger.info(f"Request data keys: {list(data.keys())}")
        logger.info(f"Request data (truncated): {str(data)[:500]}...")
            
        model_id = data.get('model', self.api.current_model)
        logger.info(f"Requested model: {model_id}")
        logger.info(f"Current model: {self.api.current_model}")
        logger.info(f"Available models: {list(self.api.models.keys())}")
        
        if not model_id or model_id not in self.api.models:
            logger.error(f"Model not found: {model_id}")
            return jsonify({'error': 'Model not found'}), 404
            
        model = self.api.models[model_id]
        logger.info(f"Model type: {type(model).__name__}")
        
        # Check if model supports activation capture
        if not hasattr(model, 'activation_hook_manager'):
            logger.error(f"Model {model_id} does not support activation capture")
            return jsonify({'error': 'Model does not support activation capture'}), 400
            
        logger.info(f"Model {model_id} supports activation capture")
            
        try:
            # Extract parameters
            prompt = data.get('prompt', '')
            intervention_specs = data.get('interventions', [])
            baseline_prompt = data.get('baseline_prompt')
            
            logger.info(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
            logger.info(f"Prompt length: {len(prompt)}")
            logger.info(f"Intervention specs count: {len(intervention_specs)}")
            logger.info(f"Baseline prompt: {baseline_prompt[:100] if baseline_prompt and len(baseline_prompt) > 100 else baseline_prompt}")
            
            if not prompt:
                logger.error("Missing prompt parameter")
                return jsonify({'error': 'prompt is required'}), 400
                
            if not intervention_specs:
                logger.error("Missing interventions parameter")
                return jsonify({'error': 'interventions are required'}), 400
                
            logger.info(f"Intervention specs: {intervention_specs}")
                
            # Create activation patcher
            logger.info("Creating ActivationPatcher instance")
            patcher = ActivationPatcher(model)
            
            # Convert intervention specs
            interventions = []
            logger.info(f"Converting {len(intervention_specs)} intervention specifications")
            for i, spec in enumerate(intervention_specs):
                logger.info(f"Processing intervention {i+1}: layer={spec.get('layer_name')}, component={spec.get('component', 'residual')}, type={spec.get('type', 'zero_ablation')}")
                intervention = InterventionSpec(
                    layer_name=spec['layer_name'],
                    component=spec.get('component', 'residual'),
                    intervention_type=InterventionType(spec.get('type', 'zero_ablation')),
                    strength=spec.get('strength', 1.0),
                    target_positions=spec.get('target_positions')
                )
                interventions.append(intervention)
                
            logger.info(f"Successfully created {len(interventions)} intervention objects")
                
            # Apply interventions
            logger.info("Starting intervention application")
            start_time = time.time()
            results = patcher.apply_interventions(prompt, interventions, baseline_prompt)
            execution_time = time.time() - start_time
            
            logger.info(f"Intervention application completed in {execution_time:.3f}s")
            logger.info(f"Results type: {type(results)}")
            logger.info(f"Results keys: {list(results.keys()) if isinstance(results, dict) else 'Not a dict'}")
            logger.info("=== ACTIVATION_PATCH SUCCESS ===")
            
            return jsonify({
                'success': True,
                'results': results,
                'intervention_count': len(interventions),
                'execution_time': execution_time
            })
            
        except Exception as e:
            error_details = {
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__,
                'model_id': model_id,
                'traceback': traceback.format_exc()
            }
            logger.error(f"Activation patching failed: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            logger.error("=== ACTIVATION_PATCH ERROR ===")
            return jsonify(error_details), 500


def register_activation_routes(app, api_instance):
    """Register activation-related routes with the Flask app."""
    endpoint = ActivationEndpoint(api_instance)
    
    # Activation hooks management
    app.add_url_rule(
        '/v1/activations/hooks',
        'manage_activation_hooks',
        endpoint.manage_activation_hooks,
        methods=['POST', 'DELETE']
    )
    
    # Residual stream analysis
    app.add_url_rule(
        '/analyze/residual',
        'analyze_residual',
        endpoint.analyze_residual,
        methods=['POST']
    )
    
    # Attention analysis
    app.add_url_rule(
        '/v1/analyze/attention',
        'analyze_attention',
        endpoint.analyze_attention,
        methods=['POST']
    )
    
    # Chat completions with activations
    app.add_url_rule(
        '/v1/chat/completions/with_activations',
        'chat_completions_with_activations',
        endpoint.chat_completions_with_activations,
        methods=['POST']
    )
    
    # Activation patching
    app.add_url_rule(
        '/v1/activations/patch',
        'activation_patch',
        endpoint.activation_patch,
        methods=['POST']
    )
    
    logger.info("Activation routes registered successfully")