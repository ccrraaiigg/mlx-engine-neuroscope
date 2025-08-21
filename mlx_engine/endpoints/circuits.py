from .base import EndpointBase, logger
from flask import request, jsonify
import time

# Import circuit discovery components
try:
    from mlx_engine.activation_patching import (
        ActivationPatcher, CausalTracer, GradientBasedAttribution,
        InterventionType, InterventionSpec, CausalTracingResult,
        ComponentType, create_sophisticated_circuit_discovery_pipeline
    )
    ACTIVATION_PATCHING_AVAILABLE = True
except ImportError:
    ACTIVATION_PATCHING_AVAILABLE = False

try:
    from mlx_engine.enhanced_causal_tracing import (
        EnhancedCausalTracer, EnhancedCausalResult, NoiseConfig, AttributionConfig,
        StatisticalConfig, NoiseType, AttributionMethod, CausalMediationType,
        create_enhanced_causal_discovery_pipeline
    )
    ENHANCED_CAUSAL_TRACING_AVAILABLE = True
except ImportError:
    ENHANCED_CAUSAL_TRACING_AVAILABLE = False

# Import custom exceptions
from mlx_engine.api_server import (
    ModelNotFoundError, ModelNotSupportedError, InvalidRequestError,
    ComponentNotAvailableError, CircuitDiscoveryError
)


class CircuitEndpoint(EndpointBase):
    """Handles circuit discovery endpoints."""
    
    def __init__(self, api_instance):
        super().__init__(api_instance)
    
    def discover_circuits(self):
        """Advanced circuit discovery using activation patching and causal tracing."""
        logger.info("=== /v1/circuits/discover endpoint called ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request content length: {request.content_length}")
        
        # Check component availability
        if not ACTIVATION_PATCHING_AVAILABLE:
            logger.error("Activation patching not available")
            raise ComponentNotAvailableError(
                'activation_patching', 
                'Activation patching components not imported'
            )
        
        try:
            # Validate request data
            logger.info("Attempting to parse JSON request data...")
            data = request.get_json()
            logger.info(f"Received request data: {data}")
            
            if not data:
                raise InvalidRequestError("Request body is required", "body")
            
            # Validate required fields
            required_fields = ['prompt', 'phenomenon']
            for field in required_fields:
                if field not in data:
                    raise InvalidRequestError(f"Missing required field: {field}", field)
            
            # Check if model is loaded
            model_id = data.get('model', self.api.current_model)
            logger.info(f"Requested model: {model_id}, Current model: {self.api.current_model}")
            logger.info(f"Available models: {list(self.api.models.keys())}")
            
            if not model_id or model_id not in self.api.models:
                raise ModelNotFoundError(model_id, list(self.api.models.keys()))
            
            model = self.api.models[model_id]
            logger.info(f"Model type: {type(model)}")
            
            if not hasattr(model, 'activation_hook_manager'):
                raise ModelNotSupportedError(model_id, 'activation_hook_manager')
            
            # Validate prompt
            prompt = data.get('prompt', '')
            logger.info(f"Circuit discovery prompt: '{prompt}'")
            
            if not prompt:
                raise InvalidRequestError("Prompt is required", "prompt", prompt)
            
            # Parse configuration parameters
            logger.info("Parsing configuration parameters...")
            target_layers = data.get('target_layers', [f'model.layers.{i}' for i in range(0, 20, 4)])
            target_components = [ComponentType(comp) for comp in data.get('target_components', ['attention', 'mlp'])]
            intervention_types = [InterventionType(interv) for interv in data.get('intervention_types', ['zero_ablation'])]
            max_tokens = data.get('max_tokens', 50)
            
            logger.info(f"Target layers: {target_layers}")
            logger.info(f"Target components: {[c.value for c in target_components]}")
            logger.info(f"Intervention types: {[i.value for i in intervention_types]}")
            logger.info(f"Max tokens: {max_tokens}")
            
            # Create circuit discovery pipeline
            logger.info("Creating circuit discovery pipeline...")
            model_kit = model
            actual_model = model_kit.model  # Extract the actual nn.Module
            logger.info(f"Model kit type: {type(model_kit)}, Actual model type: {type(actual_model)}")
            
            patcher, tracer = create_sophisticated_circuit_discovery_pipeline(
                actual_model, model_kit
            )
            logger.info(f"Pipeline created - Patcher: {type(patcher)}, Tracer: {type(tracer)}")
            
            # Perform circuit discovery
            logger.info(f"Starting circuit discovery with {len(target_layers)} layers")
            start_time = time.time()
            
            results = tracer.causal_trace(
                prompt=prompt,
                target_layers=target_layers,
                target_components=target_components,
                intervention_types=intervention_types,
                max_tokens=max_tokens
            )
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Circuit discovery completed in {duration:.2f} seconds")
            logger.info(f"Results count: {len(results)}")
            
            # Format results for API response
            logger.info("Formatting results for API response...")
            circuits = []
            max_circuits = data.get('max_circuits', 10)
            
            for i, result in enumerate(results[:max_circuits]):
                logger.info(f"Processing result {i}: {type(result)}")
                try:
                    circuit = {
                        'circuit_id': f"{data.get('phenomenon', 'general')}_{result.layer_name}_{result.component.value}",
                        'phenomenon': data.get('phenomenon', 'general'),
                        'layer_name': result.layer_name,
                        'component': result.component.value,
                        'intervention_type': result.intervention_type.value,
                        'causal_effect': float(result.causal_effect),
                        'confidence': float(result.confidence),
                        'description': f"Circuit for {data.get('phenomenon', 'general')} in {result.layer_name} {result.component.value}"
                    }
                    circuits.append(circuit)
                    logger.info(f"Added circuit {i}: {circuit['circuit_id']} with effect {circuit['causal_effect']}")
                except Exception as format_error:
                    logger.error(f"Error formatting result {i}: {format_error}")
                    continue
            
            logger.info(f"Formatted {len(circuits)} circuits for response")
            
            response_data = {
                'success': True,
                'circuits': circuits,
                'phenomenon': data.get('phenomenon', 'general'),
                'execution_time': duration,
                'total_results': len(results)
            }
            
            logger.info(f"Circuit discovery completed successfully. Returning {len(circuits)} circuits")
            return jsonify(response_data)
            
        except Exception as e:
            # Handle unexpected errors
            import traceback
            logger.error(f"Unexpected error in circuit discovery: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def enhanced_circuit_discovery(self):
        """Enhanced circuit discovery with noise injection and statistical analysis (async)."""
        logger.info("=== /v1/circuits/enhanced-discover endpoint called ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request content length: {request.content_length}")
        
        try:
            # Check component availability
            if not ENHANCED_CAUSAL_TRACING_AVAILABLE:
                logger.error("Enhanced causal tracing not available")
                raise ComponentNotAvailableError(
                    'enhanced_causal_tracing', 
                    'Enhanced causal tracing components not imported'
                )
            
            # Validate request data
            logger.info("Attempting to parse JSON request data...")
            data = request.get_json()
            logger.info(f"Received request data: {data}")
            
            if not data:
                raise InvalidRequestError("Request body is required", "body")
            
            # Generate task ID and start background task
            task_id = self.api._generate_task_id()
            logger.info(f"Starting enhanced circuit discovery task: {task_id}")
            
            # Start the background task
            self.api._run_background_task(
                self._perform_enhanced_circuit_discovery,
                task_id,
                data
            )
            
            # Return task ID immediately
            return jsonify({
                'success': True,
                'task_id': task_id,
                'status': 'started',
                'message': 'Enhanced circuit discovery started in background. Use /v1/tasks/status/{task_id} to check progress.',
                'estimated_duration': '15-30 minutes'
            })
            
        except Exception as e:
            # Handle unexpected errors
            import traceback
            logger.error(f"Unexpected error in enhanced circuit discovery: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def _perform_enhanced_circuit_discovery(self, data):
        """Perform enhanced circuit discovery in background."""
        logger.info("Starting enhanced circuit discovery background task...")
        
        try:
            # Validate required fields
            required_fields = ['prompt']
            for field in required_fields:
                if field not in data:
                    raise InvalidRequestError(f"Missing required field: {field}", field)
            
            # Check if model is loaded
            model_id = data.get('model', self.api.current_model)
            logger.info(f"Requested model: {model_id}, Current model: {self.api.current_model}")
            logger.info(f"Available models: {list(self.api.models.keys())}")
            
            if not model_id or model_id not in self.api.models:
                raise ModelNotFoundError(model_id, list(self.api.models.keys()))
            
            model = self.api.models[model_id]
            logger.info(f"Model type: {type(model)}")
            
            if not hasattr(model, 'activation_hook_manager'):
                raise ModelNotSupportedError(model_id, 'activation_hook_manager')
            
            # Validate prompt
            prompt = data.get('prompt', '')
            logger.info(f"Enhanced circuit discovery prompt: '{prompt}'")
            
            if not prompt:
                raise InvalidRequestError("Prompt is required", "prompt", prompt)
            
            # Parse configuration parameters
            logger.info("Parsing configuration parameters...")
            noise_config = NoiseConfig(
                noise_type=NoiseType(data.get('noise_type', 'gaussian')),
                strength=data.get('noise_scale', 0.1)
            )
            logger.info(f"Noise config: {noise_config}")
            
            attribution_config = AttributionConfig(
                method=AttributionMethod(data.get('attribution_method', 'integrated_gradients')),
                steps=data.get('attribution_steps', 50),
                baseline_strategy=data.get('baseline_strategy', 'zero')
            )
            logger.info(f"Attribution config: {attribution_config}")
            
            statistical_config = StatisticalConfig(
                confidence_level=1.0 - data.get('significance_threshold', 0.05),
                multiple_testing_correction=data.get('multiple_comparisons', 'bonferroni'),
                bootstrap_samples=data.get('bootstrap_samples', 1000)
            )
            logger.info(f"Statistical config: {statistical_config}")
            
            # Create enhanced causal discovery pipeline
            logger.info("Creating enhanced causal discovery pipeline...")
            model_kit = model
            actual_model = model_kit.model  # Extract the actual nn.Module
            logger.info(f"Model kit type: {type(model_kit)}, Actual model type: {type(actual_model)}")
            
            patcher, enhanced_tracer = create_enhanced_causal_discovery_pipeline(
                actual_model, noise_config, attribution_config, statistical_config, model_kit
            )
            logger.info(f"Pipeline created - Patcher: {type(patcher)}, Enhanced tracer: {type(enhanced_tracer)}")
            
            # Perform enhanced circuit discovery using the enhanced tracer
            target_layers = [f'model.layers.{i}' for i in range(0, 20, 4)]  # Sample layers
            target_components = [ComponentType.ATTENTION, ComponentType.MLP]
            intervention_types = [InterventionType.ZERO_ABLATION, InterventionType.MEAN_ABLATION]
            
            logger.info(f"Starting enhanced causal trace with target_layers: {target_layers}")
            logger.info(f"Target components: {[c.value for c in target_components]}")
            logger.info(f"Intervention types: {[i.value for i in intervention_types]}")
            logger.info(f"About to call enhanced_causal_trace - this will trigger GPU activity")
            logger.info(f"PROGRESS: Starting enhanced circuit discovery process (estimated 15-30 minutes)")
            logger.info(f"PROGRESS: Processing {len(target_layers)} layers x {len(target_components)} components x {len(intervention_types)} intervention types")
            logger.info(f"PROGRESS: Total combinations to analyze: {len(target_layers) * len(target_components) * len(intervention_types)}")
            
            try:
                start_time = time.time()
                logger.info(f"PROGRESS: [0%] Starting enhanced causal trace at {time.strftime('%H:%M:%S')}")
                
                enhanced_results = enhanced_tracer.enhanced_causal_trace(
                    prompt=prompt,
                    target_layers=target_layers,
                    target_components=target_components,
                    intervention_types=intervention_types,
                    max_tokens=50
                )
                
                end_time = time.time()
                duration = end_time - start_time
                logger.info(f"PROGRESS: [100%] Enhanced causal trace completed successfully in {duration:.2f} seconds")
                logger.info(f"Enhanced causal trace completed successfully. Results count: {len(enhanced_results)}")
                logger.info(f"Results type: {type(enhanced_results)}")
                if enhanced_results:
                    logger.info(f"First result type: {type(enhanced_results[0])}")
                    logger.info(f"First result attributes: {dir(enhanced_results[0])}")
            except Exception as trace_error:
                logger.error(f"Error during enhanced_causal_trace: {trace_error}")
                logger.error(f"Trace error type: {type(trace_error)}")
                import traceback
                logger.error(f"Trace error traceback: {traceback.format_exc()}")
                raise CircuitDiscoveryError(f"Enhanced causal trace failed: {str(trace_error)}", "enhanced_causal_trace", trace_error)
            
            # Format results for API response
            logger.info("Formatting results for API response...")
            circuits = []
            max_circuits = data.get('max_circuits', 10)
            logger.info(f"Processing up to {max_circuits} circuits from {len(enhanced_results)} results")
            
            for i, result in enumerate(enhanced_results[:max_circuits]):
                logger.info(f"Processing result {i}: {type(result)}")
                try:
                    logger.info(f"Result {i} base_result: {result.base_result}")
                    logger.info(f"Result {i} effect_size: {result.effect_size}")
                    logger.info(f"Result {i} layer_name: {result.base_result.layer_name}")
                    logger.info(f"Result {i} component: {result.base_result.component}")
                    
                    circuit = {
                        'circuit_id': f"{data.get('phenomenon', 'general')}_{result.base_result.layer_name}_{result.base_result.component.value}",
                        'phenomenon': data.get('phenomenon', 'general'),
                        'layer_name': result.base_result.layer_name,
                        'component': result.base_result.component.value,
                        'intervention_type': result.base_result.intervention_type.value,
                        'causal_effect': float(result.effect_size),
                        'confidence': float(result.base_result.confidence),
                        'attribution_scores': self.api._safe_process_attribution_scores(result.attribution_scores) if result.attribution_scores else {},
                        'noise_robustness': self.api._safe_process_noise_robustness(result.noise_robustness) if result.noise_robustness else {},
                        'statistical_significance': result.statistical_significance or {},
                        'description': f"Enhanced circuit for {data.get('phenomenon', 'general')} in {result.base_result.layer_name} {result.base_result.component.value}"
                    }
                    circuits.append(circuit)
                    logger.info(f"Added circuit {i}: {circuit['circuit_id']} with effect {circuit['causal_effect']}")
                except Exception as format_error:
                    logger.error(f"Error formatting result {i}: {format_error}")
                    logger.error(f"Result {i} raw data: {result}")
                    continue
            
            logger.info(f"Formatted {len(circuits)} circuits for response")
            
            # Create result object
            logger.info("Creating result object...")
            try:
                result = type('EnhancedResult', (), {
                    'circuits': circuits,
                    'statistical_analysis': enhanced_results[0].statistical_significance if enhanced_results else {},
                    'noise_robustness': enhanced_results[0].noise_robustness if enhanced_results else {},
                    'attribution_scores': enhanced_results[0].attribution_scores if enhanced_results else {},
                    'mediation_analysis': enhanced_results[0].mediation_analysis if enhanced_results else {},
                    'confidence_intervals': enhanced_results[0].uncertainty_bounds if enhanced_results else {}
                })()
                logger.info("Result object created successfully")
                
                response_data = {
                    'success': True,
                    'enhanced_circuits': result.circuits,
                    'statistical_analysis': result.statistical_analysis,
                    'noise_robustness': self.api._safe_process_noise_robustness(result.noise_robustness) if result.noise_robustness else {},
                    'attribution_scores': self.api._safe_process_attribution_scores(result.attribution_scores) if result.attribution_scores else {},
                    'mediation_analysis': self.api._safe_process_mediation_analysis(result.mediation_analysis) if result.mediation_analysis else {},
                    'confidence_intervals': result.confidence_intervals,
                    'phenomenon': data.get('phenomenon', 'general')
                }
                logger.info(f"Enhanced circuit discovery completed successfully. Returning {len(result.circuits)} circuits")
                logger.info(f"Response data keys: {list(response_data.keys())}")
                logger.info(f"Response data size estimate: {len(str(response_data))} characters")
                
                return response_data
                
            except Exception as response_error:
                logger.error(f"Error creating response: {response_error}")
                logger.error(f"Enhanced results available: {len(enhanced_results) if enhanced_results else 0}")
                logger.error(f"Circuits formatted: {len(circuits)}")
                import traceback
                logger.error(f"Response error traceback: {traceback.format_exc()}")
                raise CircuitDiscoveryError(f"Failed to create response: {str(response_error)}", "response_creation", response_error)
            
        except Exception as e:
            # Handle unexpected errors
            import traceback
            logger.error(f"Unexpected error in enhanced circuit discovery: {e}")
            logger.error(f"Exception type: {type(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e


def register_circuit_routes(app, api_instance):
    """Register circuit discovery routes with the Flask app."""
    circuit_endpoint = CircuitEndpoint(api_instance)
    
    @app.route('/v1/circuits/discover', methods=['POST'])
    def discover_circuits():
        return circuit_endpoint.discover_circuits()
    
    @app.route('/v1/circuits/enhanced-discover', methods=['POST'])
    def enhanced_circuit_discovery():
        return circuit_endpoint.enhanced_circuit_discovery()