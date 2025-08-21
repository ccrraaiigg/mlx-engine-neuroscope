"""Health endpoint module."""

import datetime
from flask import jsonify
from .base import EndpointBase

class HealthEndpoint(EndpointBase):
    """Health check endpoint implementation."""
    
    def health(self):
        """Health check endpoint with version and timestamp."""
        self.logger.info(f"=== HEALTH CHECK REQUEST ===")
        self.logger.info(f"Current model: {self.api.current_model}")
        self.logger.info(f"Loaded models: {list(self.api.models.keys())}")
        self.logger.info(f"Total loaded models: {len(self.api.models)}")
        self.logger.info(f"Model ready status: {self.api.current_model is not None}")
        
        response_data = {
            'status': 'healthy', 
            'service': 'mlx-engine-neuroscope',
            'component': 'MLX Engine REST API',
            'version': '1.2.0',
            'timestamp': datetime.datetime.now().isoformat(),
            'current_model': self.api.current_model,
            'ready': self.api.current_model is not None,
            'loaded_models': list(self.api.models.keys()),
            'total_loaded': len(self.api.models)
        }
        
        self.logger.info(f"Health check response: {response_data}")
        self.logger.info(f"=== END HEALTH CHECK ===")
        return jsonify(response_data)

def register_health_routes(api_instance):
    """Register health routes with the Flask app."""
    health_endpoint = HealthEndpoint(api_instance)
    
    api_instance.app.route('/health', methods=['GET'])(health_endpoint.health)