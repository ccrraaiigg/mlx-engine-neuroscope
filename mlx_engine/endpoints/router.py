from .health import register_health_routes
from .tasks import register_task_routes
from .models import register_model_routes
from .generation import register_generation_routes
from .activations import register_activation_routes
from .circuits import register_circuit_routes

# Import feature localization if available
try:
    from mlx_engine.feature_localization import (
        FeatureLocalizer, FeatureSpec, LocalizedFeature, FeatureLocalizationResult,
        FeatureType, LocalizationMethod, SparseAutoencoder, DictionaryLearner, 
        ProbingClassifier, COMMON_FEATURES, create_feature_localization_pipeline
    )
    FEATURE_LOCALIZATION_AVAILABLE = True
except ImportError:
    FEATURE_LOCALIZATION_AVAILABLE = False

from .base import logger
from flask import request, jsonify


def register_feature_routes(app, api_instance):
    """Register feature localization routes if available."""
    if not FEATURE_LOCALIZATION_AVAILABLE:
        logger.warning("Feature localization not available, skipping feature routes")
        return
    
    @app.route('/v1/features/localize', methods=['POST'])
    def localize_features():
        """Localize specific features in neural activations."""
        logger.info("=== /v1/features/localize endpoint called ===")
        logger.info(f"Request method: {request.method}")
        logger.info(f"Request headers: {dict(request.headers)}")
        logger.info(f"Request content type: {request.content_type}")
        logger.info(f"Request content length: {request.content_length}")
        
        if FeatureLocalizer is None:
            logger.error("FeatureLocalizer not available")
            return jsonify({'error': 'Feature localization not available'}), 501
        
        try:
            logger.info("Attempting to parse JSON request data...")
            data = request.get_json()
            logger.info(f"Request data keys: {list(data.keys()) if data else 'None'}")
            logger.info(f"Request data (truncated): {str(data)[:500]}..." if data and len(str(data)) > 500 else f"Request data: {data}")
            
            # Validate required fields
            if 'features' not in data:
                logger.error("Missing required field: features")
                return jsonify({'error': 'Missing required field: features'}), 400
            
            logger.info(f"Features count: {len(data['features'])}")
            
            # Check if model is loaded
            model_id = data.get('model', api_instance.current_model)
            logger.info(f"Requested model: {model_id}, Current model: {api_instance.current_model}")
            logger.info(f"Available models: {list(api_instance.models.keys())}")
            
            if not model_id or model_id not in api_instance.models:
                logger.error(f"Model not found: {model_id}")
                return jsonify({'error': 'Model not found'}), 404
            
            model = api_instance.models[model_id]
            logger.info(f"Model type: {type(model)}")
            
            # Parse feature specifications
            logger.info("Parsing feature specifications...")
            feature_specs = []
            for i, feature_data in enumerate(data['features']):
                try:
                    logger.info(f"Processing feature {i+1}: {feature_data.get('name', 'unnamed')}")
                    logger.info(f"Feature type: {feature_data.get('feature_type', 'semantic')}")
                    logger.info(f"Target layers: {feature_data.get('target_layers', ['model.layers.12'])}")
                    logger.info(f"Examples count: {len(feature_data.get('examples', []))}")
                    logger.info(f"Counter examples count: {len(feature_data.get('counter_examples', []))}")
                    
                    feature_spec = FeatureSpec(
                        name=feature_data['name'],
                        feature_type=FeatureType(feature_data.get('feature_type', 'semantic')),
                        description=feature_data.get('description', ''),
                        target_layers=feature_data.get('target_layers', ['model.layers.12']),
                        examples=feature_data.get('examples', []),
                        counter_examples=feature_data.get('counter_examples', [])
                    )
                    feature_specs.append(feature_spec)
                except Exception as e:
                    logger.error(f"Invalid feature specification {i+1}: {e}")
                    return jsonify({'error': f'Invalid feature specification: {e}'}), 400
            
            # Get localization method
            method_str = data.get('method', 'sparse_autoencoder')
            logger.info(f"Localization method: {method_str}")
            logger.info(f"Additional parameters: {data.get('parameters', {})}")
            
            try:
                method = LocalizationMethod(method_str)
            except ValueError:
                logger.error(f"Invalid localization method: {method_str}")
                return jsonify({'error': f'Invalid localization method: {method_str}'}), 400
            
            # Create feature localizer
            logger.info("Creating feature localization pipeline...")
            localizer = create_feature_localization_pipeline(model)
            logger.info(f"Feature localizer created: {type(localizer)}")
            
            # Perform feature localization
            logger.info("Starting feature localization...")
            import time
            start_time = time.time()
            
            result = localizer.localize_features(
                feature_specs=feature_specs,
                method=method,
                **data.get('parameters', {})
            )
            
            end_time = time.time()
            execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
            logger.info(f"Feature localization completed in {execution_time:.2f}ms")
            logger.info(f"Result type: {type(result)}")
            logger.info(f"Features found: {len(result.features)}")
            
            # Convert result to JSON-serializable format
            logger.info("Converting results to JSON format...")
            features_json = []
            for i, feature in enumerate(result.features):
                logger.info(f"Processing feature result {i+1}: {feature.feature_spec.name}")
                logger.info(f"  Layer: {feature.layer_name}, Confidence: {feature.confidence}")
                logger.info(f"  Activation strength: {feature.activation_strength}")
                logger.info(f"  Neuron indices count: {len(feature.neuron_indices)}")
                
                features_json.append({
                    'feature_name': feature.feature_spec.name,
                    'feature_type': feature.feature_spec.feature_type.value,
                    'layer_name': feature.layer_name,
                    'neuron_indices': feature.neuron_indices,
                    'activation_strength': feature.activation_strength,
                    'confidence': feature.confidence,
                    'localization_method': feature.localization_method.value,
                    'metadata': feature.metadata
                })
            
            logger.info(f"Returning response with {len(features_json)} features")
            logger.info(f"Layer analysis keys: {list(result.layer_analysis.keys()) if hasattr(result, 'layer_analysis') and result.layer_analysis else 'None'}")
            logger.info(f"Feature interactions count: {len(result.feature_interactions) if hasattr(result, 'feature_interactions') and result.feature_interactions else 0}")
            
            return jsonify({
                'features': features_json,
                'layer_analysis': result.layer_analysis,
                'feature_interactions': result.feature_interactions,
                'execution_time_ms': result.execution_time_ms,
                'total_execution_time': execution_time,
                'metadata': result.metadata
            })
        
        except Exception as e:
            import traceback
            error_details = {
                'error_type': type(e).__name__,
                'model_id': data.get('model', api_instance.current_model) if 'data' in locals() else 'unknown',
                'features_count': len(data.get('features', [])) if 'data' in locals() and data else 0,
                'method': data.get('method', 'unknown') if 'data' in locals() and data else 'unknown',
                'traceback': traceback.format_exc()
            }
            logger.error(f"Feature localization failed: {e}")
            logger.error(f"Error details: {error_details}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify({
                'error': str(e),
                'error_details': error_details
            }), 500
    
    @app.route('/v1/features/common', methods=['GET'])
    def get_common_features():
        """Get predefined common feature specifications."""
        if FeatureLocalizer is None:
            return jsonify({'error': 'Feature localization not available'}), 501
        
        try:
            common_features_json = []
            for feature_spec in COMMON_FEATURES:
                common_features_json.append({
                    'name': feature_spec.name,
                    'feature_type': feature_spec.feature_type.value,
                    'description': feature_spec.description,
                    'target_layers': feature_spec.target_layers,
                    'examples': feature_spec.examples,
                    'counter_examples': feature_spec.counter_examples
                })
            
            return jsonify({
                'common_features': common_features_json,
                'total_count': len(COMMON_FEATURES)
            })
        
        except Exception as e:
            logger.error(f"Failed to get common features: {e}")
            return jsonify({'error': str(e)}), 500
    
    @app.route('/v1/features/methods', methods=['GET'])
    def get_localization_methods():
        """Get available feature localization methods."""
        if FeatureLocalizer is None:
            return jsonify({'error': 'Feature localization not available'}), 501
        
        try:
            methods = []
            for method in LocalizationMethod:
                method_info = {
                    'name': method.value,
                    'description': _get_method_description(method)
                }
                methods.append(method_info)
            
            return jsonify({
                'methods': methods,
                'default_method': 'sparse_autoencoder'
            })
        
        except Exception as e:
            logger.error(f"Failed to get localization methods: {e}")
            return jsonify({'error': str(e)}), 500


def _get_method_description(method):
    """Get description for a localization method."""
    descriptions = {
        LocalizationMethod.SPARSE_AUTOENCODER: "Uses sparse autoencoders to identify feature-specific neurons",
        LocalizationMethod.DICTIONARY_LEARNING: "Applies dictionary learning to decompose activations into interpretable features",
        LocalizationMethod.PCA: "Uses Principal Component Analysis to find the most important dimensions",
        LocalizationMethod.PROBING_CLASSIFIER: "Trains linear probes to identify neurons that encode specific features",
        LocalizationMethod.GRADIENT_ATTRIBUTION: "Uses gradient-based attribution to find neurons most relevant to features"
    }
    return descriptions.get(method, "Unknown method")


def register_all_routes(app, api_instance):
    """Register all endpoint routes with the Flask app."""
    logger.info("Registering all endpoint routes...")
    
    # Register core endpoints
    register_health_routes(api_instance)
    register_task_routes(api_instance)
    register_model_routes(app, api_instance)
    register_generation_routes(app, api_instance)
    register_activation_routes(app, api_instance)
    register_circuit_routes(app, api_instance)
    
    # Register optional feature localization routes
    register_feature_routes(app, api_instance)
    
    logger.info("All endpoint routes registered successfully")