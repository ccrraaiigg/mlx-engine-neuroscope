#!/usr/bin/env python3
"""
MLX Engine API Server

A Flask-based API server for the MLX Engine with modular endpoint organization.
"""

from typing import Dict, List, Optional, Any, Iterator
import json
import asyncio
import os
from pathlib import Path
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor

# Configure logging
log_file_path = Path(__file__).parent / 'api_server.log'
if log_file_path.exists():
    log_file_path.unlink()  # Delete old log file

logging.basicConfig(
    level=logging.DEBUG,  # More verbose logging
    format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler(str(log_file_path))
    ],
    force=True  # Override any existing configuration
)
logger = logging.getLogger(__name__)
logger.info(f"API Server logging initialized. Log file: {log_file_path}")

# Custom exceptions
class MLXEngineAPIError(Exception):
    """Base exception for MLX Engine API errors."""
    def __init__(self, message: str, status_code: int = 500, details: dict = None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}

class ModelNotFoundError(MLXEngineAPIError):
    """Raised when a requested model is not found."""
    def __init__(self, model_id: str, available_models: list = None):
        super().__init__(f"Model '{model_id}' not found", 404, {
            'model_id': model_id, 'available_models': available_models or []
        })

class ModelNotSupportedError(MLXEngineAPIError):
    """Raised when a model doesn't support a required feature."""
    def __init__(self, model_id: str, required_feature: str):
        super().__init__(f"Model '{model_id}' does not support {required_feature}", 400, {
            'model_id': model_id, 'required_feature': required_feature
        })

class InvalidRequestError(MLXEngineAPIError):
    """Raised when request parameters are invalid."""
    def __init__(self, message: str, field: str = None, value=None):
        super().__init__(message, 400, {'field': field, 'value': value})

class ComponentNotAvailableError(MLXEngineAPIError):
    """Raised when a required component is not available."""
    def __init__(self, component_name: str, reason: str = None):
        message = f"Component '{component_name}' is not available"
        if reason:
            message += f": {reason}"
        super().__init__(message, 503, {'component': component_name, 'reason': reason})

class CircuitDiscoveryError(MLXEngineAPIError):
    """Raised when circuit discovery operations fail."""
    def __init__(self, message: str, stage: str = None, original_error: Exception = None):
        super().__init__(message, 500, {
            'stage': stage, 'original_error': str(original_error) if original_error else None
        })

# Flask imports
try:
    from flask import Flask, request, jsonify, Response, stream_template
    from flask_cors import CORS
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

# MLX Engine imports
from mlx_engine import (
    load_model, 
    create_generator, 
    create_generator_with_activations,
    tokenize
)
from mlx_engine.model_kit.model_kit import ModelKit
from mlx_engine.vision_model_kit.vision_model_kit import VisionModelKit

# Activation hooks
try:
    logger.info("Attempting to import from activation_hooks...")
    from mlx_engine.activation_hooks import serialize_activations, ActivationHookSpec
    logger.info(f"Successfully imported ActivationHookSpec: {ActivationHookSpec}")
except ImportError as e:
    logger.error(f"Failed to import from activation_hooks: {e}")
    raise

# Activation patching
try:
    from mlx_engine.activation_patching import (
        ActivationPatcher, CausalTracer, GradientBasedAttribution,
        InterventionType, InterventionSpec, CausalTracingResult,
        ComponentType, create_sophisticated_circuit_discovery_pipeline
    )
    logger.info("Successfully imported activation patching components")
except ImportError as e:
    logger.error(f"Failed to import activation patching: {e}")
    ActivationPatcher = None

# Feature localization
try:
    from mlx_engine.feature_localization import (
        FeatureLocalizer, FeatureSpec, LocalizedFeature, FeatureLocalizationResult,
        FeatureType, LocalizationMethod, SparseAutoencoder, DictionaryLearner, 
        ProbingClassifier, COMMON_FEATURES, create_feature_localization_pipeline
    )
    logger.info("Successfully imported feature localization components")
except ImportError as e:
    logger.warning(f"Feature localization not available: {e}")

# Enhanced causal tracing
try:
    from mlx_engine.enhanced_causal_tracing import (
        EnhancedCausalTracer, EnhancedCausalResult, NoiseConfig, AttributionConfig,
        StatisticalConfig, NoiseType, AttributionMethod, CausalMediationType,
        create_enhanced_causal_discovery_pipeline
    )
    ENHANCED_CAUSAL_TRACING_AVAILABLE = True
    logger.info("Successfully imported enhanced causal tracing components")
except ImportError as e:
    logger.warning(f"Enhanced causal tracing not available: {e}")
    ENHANCED_CAUSAL_TRACING_AVAILABLE = False

# Attention analysis
try:
    from mlx_engine.attention_analysis import (
        AttentionAnalyzer, AttentionAnalysisResult, AttentionPatternType,
        AttentionScope, AttentionHead, AttentionPattern, CrossLayerDependency,
        create_attention_analysis_pipeline
    )
    ATTENTION_ANALYSIS_AVAILABLE = True
    logger.info("Successfully imported attention analysis components")
except ImportError as e:
    logger.warning(f"Attention analysis not available: {e}")
    ATTENTION_ANALYSIS_AVAILABLE = False

# Import modular router
from mlx_engine.endpoints.router import register_all_routes


class MLXEngineAPI:
    """Main API server class for MLX Engine."""
    
    def __init__(self):
        """Initialize the MLX Engine API server."""
        if not FLASK_AVAILABLE:
            raise ImportError("Flask is required but not available")
            
        self.app = Flask(__name__)
        CORS(self.app)
        
        # Initialize state
        self.models = {}
        self.current_model = None
        self.model_kit = None  # Will be initialized when a model is loaded
        self.vision_model_kit = None  # Will be initialized when a vision model is loaded
        self.activation_hooks = {}
        self.running_tasks = {}
        self.task_results = {}
        self.task_lock = threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Setup routes and logging
        self._setup_request_logging()
        self._setup_routes()
        
        logger.info("MLX Engine API initialized successfully")
    
    def _generate_task_id(self):
        """Generate a unique task ID."""
        import uuid
        return str(uuid.uuid4())
    
    def _run_background_task(self, task_func, task_id, *args, **kwargs):
        """Run a task in the background."""
        try:
            with self.task_lock:
                self.running_tasks[task_id] = {
                    'started_at': time.time(),
                    'description': kwargs.get('description', 'Background task')
                }
            
            result = task_func(*args, **kwargs)
            
            with self.task_lock:
                self.running_tasks.pop(task_id, None)
                self.task_results[task_id] = {
                    'status': 'completed',
                    'result': result,
                    'completed_at': time.time()
                }
        except Exception as e:
            logger.error(f"Background task {task_id} failed: {e}")
            with self.task_lock:
                self.running_tasks.pop(task_id, None)
                self.task_results[task_id] = {
                    'status': 'failed',
                    'error': str(e),
                    'completed_at': time.time()
                }
    
    def _get_directory_size(self, directory_path):
        """Calculate the size of a directory in MB."""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(directory_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return round(total_size / (1024 * 1024), 2)  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not calculate directory size for {directory_path}: {e}")
            return 0
    
    def _setup_request_logging(self):
        """Setup request logging middleware."""
        @self.app.before_request
        def log_request_info():
            logger.info(f"Request: {request.method} {request.url}")
            if request.method in ['POST', 'PUT', 'PATCH'] and request.is_json:
                try:
                    logger.debug(f"Request data: {request.get_json()}")
                except Exception as e:
                    logger.debug(f"Could not parse JSON data: {e}")
        
        @self.app.after_request
        def log_response_info(response):
            logger.info(f"Response: {response.status_code}")
            return response
        
        @self.app.errorhandler(MLXEngineAPIError)
        def handle_api_error(error):
            logger.error(f"API Error: {error.message}")
            return jsonify({
                'error': error.message,
                'status_code': error.status_code,
                'details': error.details
            }), error.status_code
        
        @self.app.errorhandler(Exception)
        def handle_unexpected_error(error):
            logger.error(f"Unexpected error: {str(error)}", exc_info=True)
            return jsonify({
                'error': 'Internal server error',
                'message': str(error),
                'details': {'unexpected_error': True}
            }), 500
    
    def _setup_routes(self):
        """Set up API routes using modular structure."""
        logger.info("Registering API routes using modular structure...")
        register_all_routes(self.app, self)
    
    def run(self, host='0.0.0.0', port=8000, debug=False):
        """Run the API server."""
        logger.info(f"Starting MLX Engine API server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)


def create_app():
    """Factory function to create Flask app."""
    api = MLXEngineAPI()
    return api.app


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='MLX Engine API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    
    api = MLXEngineAPI()
    api.run(host=args.host, port=args.port, debug=args.debug)
