from flask import request, jsonify
from pathlib import Path
import os
import glob
from mlx_engine import load_model
from .base import EndpointBase


class ModelEndpoint(EndpointBase):
    """Handles model management endpoints."""
    
    def __init__(self, api_instance):
        super().__init__(api_instance)
    
    def list_models(self):
        """List all loaded models."""
        self.logger.info("=== LIST MODELS REQUEST ===")
        
        try:
            models_info = []
            for model_id, model in self.api.models.items():
                model_info = {
                    'model_id': model_id,
                    'status': 'loaded',
                    'supports_activations': hasattr(model, 'activation_hook_manager'),
                    'is_current': model_id == self.api.current_model,
                    'model_type': type(model).__name__
                }
                models_info.append(model_info)
                self.logger.info(f"Model info: {model_info}")
            
            response_data = {
                'models': models_info,
                'current_model': self.api.current_model,
                'total_models': len(models_info)
            }
            
            self.logger.info(f"List models response: {response_data}")
            self.logger.info("=== END LIST MODELS ===")
            return jsonify(self.create_success_response(response_data))
            
        except Exception as e:
            self.logger.error(f"Failed to list models: {e}")
            return jsonify(self.create_error_response(e)), 500
    
    def list_available_models(self):
        """List all available models that can be loaded."""
        self.logger.info("=== AVAILABLE MODELS REQUEST ===")
        
        try:
            # Define search paths for models
            model_search_paths = [
                os.path.expanduser("~/.cache/huggingface/hub"),
                os.path.expanduser("~/models"),
                "/opt/models",
                "./models",
                os.path.expanduser("~/.mlx/models"),
                os.path.expanduser("~/.lmstudio/models/nightmedia")
            ]
            
            # Define patterns to look for model directories
            model_patterns = [
                "models--*",
                "*-*-*",  # Common HF model naming
                "*/*",     # Subdirectories
                "*"        # Any directory
            ]
            
            available_models = []
            found_paths = set()
            
            # Search for models in common locations
            for search_path in model_search_paths:
                self.logger.info(f"Checking search path: {search_path}")
                if os.path.exists(search_path):
                    self.logger.info(f"Search path exists: {search_path}")
                    for pattern in model_patterns:
                        self.logger.info(f"Searching with pattern: {pattern}")
                        search_pattern = os.path.join(search_path, pattern)
                        self.logger.info(f"Full search pattern: {search_pattern}")
                        
                        for model_path in glob.glob(search_pattern, recursive=True):
                            self.logger.info(f"Found potential model path: {model_path}")
                            if os.path.isdir(model_path) and model_path not in found_paths:
                                self.logger.info(f"Checking directory: {model_path}")
                                # Check if it looks like a valid model directory
                                model_files = os.listdir(model_path)
                                self.logger.info(f"Files in {model_path}: {model_files[:5]}{'...' if len(model_files) > 5 else ''}")
                                
                                has_model_files = any(
                                    f.endswith('.bin') or f.endswith('.safetensors') or 
                                    f == 'config.json' or f == 'tokenizer.json'
                                    for f in model_files
                                )
                                
                                self.logger.info(f"Has model files: {has_model_files}")
                                if has_model_files:
                                    model_name = Path(model_path).name
                                    model_info = {
                                        'model_id': model_name,
                                        'model_path': model_path,
                                        'size_mb': self.api._get_directory_size(model_path),
                                        'files': model_files[:10]  # First 10 files for reference
                                    }
                                    available_models.append(model_info)
                                    found_paths.add(model_path)
                                    self.logger.info(f"Added available model: {model_name} at {model_path}")
                                else:
                                    self.logger.info(f"Skipping {model_path} - not directory or already found")
                else:
                    self.logger.warning(f"Search path does not exist: {search_path}")
            
            self.logger.info(f"Found {len(available_models)} available models")
            
            # Also include currently loaded models
            loaded_models = []
            for model_id, model in self.api.models.items():
                model_info = {
                    'model_id': model_id,
                    'status': 'loaded',
                    'supports_activations': hasattr(model, 'activation_hook_manager'),
                    'is_current': model_id == self.api.current_model,
                    'model_type': type(model).__name__
                }
                loaded_models.append(model_info)
                self.logger.info(f"Loaded model info: {model_info}")
            
            response_data = {
                'available_models': available_models,
                'loaded_models': loaded_models,
                'current_model': self.api.current_model,
                'search_paths': model_search_paths,
                'total_available': len(available_models),
                'total_loaded': len(loaded_models)
            }
            
            self.logger.info(f"Available models response: {len(available_models)} available, {len(loaded_models)} loaded")
            self.logger.info("=== END AVAILABLE MODELS ===")
            return jsonify(self.create_success_response(response_data))
            
        except Exception as e:
            import traceback
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
            self.logger.error(f"Failed to get available models: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify(self.create_error_response(Exception(error_details['error']))), 500
    
    def load_model(self):
        """Load a model from path."""
        self.logger.info("=== LOAD MODEL REQUEST ===")
        data = request.get_json()
        self.logger.info(f"Request data: {data}")
        
        if not data or 'model_path' not in data:
            self.logger.error("Missing model_path in request data")
            return jsonify(self.create_error_response(Exception('model_path is required'))), 400
        
        model_path = data['model_path']
        model_id = data.get('model_id', Path(model_path).name)
        
        self.logger.info(f"Loading model from path: {model_path}")
        self.logger.info(f"Model ID: {model_id}")
        self.logger.info(f"Current loaded models before: {list(self.api.models.keys())}")
        self.logger.info(f"Current active model before: {self.api.current_model}")
        
        # Log all loading parameters
        load_params = {
            'vocab_only': data.get('vocab_only', False),
            'max_kv_size': data.get('max_kv_size', 4096),
            'trust_remote_code': data.get('trust_remote_code', False),
            'kv_bits': data.get('kv_bits'),
            'kv_group_size': data.get('kv_group_size'),
            'quantized_kv_start': data.get('quantized_kv_start')
        }
        self.logger.info(f"Load parameters: {load_params}")
        
        try:
            self.logger.info("Starting model loading process...")
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
            
            self.logger.info(f"Model loaded successfully. Type: {type(model).__name__}")
            self.logger.info(f"Model supports activations: {hasattr(model, 'activation_hook_manager')}")
            
            self.api.models[model_id] = model
            self.api.current_model = model_id
            
            # Initialize model_kit for the current model
            from mlx_engine.model_kit.model_kit import ModelKit
            try:
                self.api.model_kit = ModelKit(Path(model_path))
                self.logger.info(f"ModelKit initialized successfully for {model_id}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize ModelKit: {e}")
                self.api.model_kit = None
            
            self.logger.info("Model registered in models dict")
            self.logger.info(f"Current loaded models after: {list(self.api.models.keys())}")
            self.logger.info(f"Current active model after: {self.api.current_model}")
            
            response_data = {
                'model_id': model_id,
                'status': 'loaded',
                'supports_activations': hasattr(model, 'activation_hook_manager'),
                'model_type': type(model).__name__,
                'total_loaded_models': len(self.api.models)
            }
            
            self.logger.info(f"Load model response: {response_data}")
            self.logger.info("=== END LOAD MODEL SUCCESS ===")
            return jsonify(self.create_success_response(response_data))
            
        except Exception as e:
            import traceback
            error_details = {
                'error': str(e),
                'error_type': type(e).__name__,
                'model_path': model_path,
                'model_id': model_id,
                'traceback': traceback.format_exc(),
                'provided_parameters': {
                    'vocab_only': data.get('vocab_only', False),
                    'max_kv_size': data.get('max_kv_size', 4096),
                    'trust_remote_code': data.get('trust_remote_code', False),
                    'kv_bits': data.get('kv_bits'),
                    'kv_group_size': data.get('kv_group_size'),
                    'quantized_kv_start': data.get('quantized_kv_start')
                }
            }
            self.logger.error(f"Failed to load model {model_id} from {model_path}: {e}")
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return jsonify(self.create_error_response(Exception(error_details['error']))), 500


def register_model_routes(app, api_instance):
    """Register model management routes."""
    endpoint = ModelEndpoint(api_instance)
    
    app.add_url_rule('/v1/models', 'list_models', endpoint.list_models, methods=['GET'])
    app.add_url_rule('/v1/models/available', 'list_available_models', endpoint.list_available_models, methods=['GET'])
    app.add_url_rule('/v1/models/load', 'load_model', endpoint.load_model, methods=['POST'])