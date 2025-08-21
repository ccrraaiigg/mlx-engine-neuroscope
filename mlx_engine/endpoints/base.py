"""Base utilities for endpoint modules."""

import logging
from flask import request, jsonify
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class EndpointBase:
    """Base class for endpoint modules with common utilities."""
    
    def __init__(self, api_instance):
        self.api = api_instance
        self.app = api_instance.app
        self.logger = logger
    
    def log_request_info(self, endpoint_name: str):
        """Log common request information."""
        self.logger.info(f"=== {endpoint_name} endpoint called ===")
        self.logger.info(f"Request method: {request.method}")
        self.logger.info(f"Request headers: {dict(request.headers)}")
        self.logger.info(f"Request content type: {request.content_type}")
        self.logger.info(f"Request content length: {request.content_length}")
    
    def create_error_response(self, error: Exception, status_code: int = 500) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            'error': {
                'type': type(error).__name__,
                'message': str(error),
                'status_code': status_code
            }
        }
    
    def create_success_response(self, data: Any, message: Optional[str] = None) -> Dict[str, Any]:
        """Create standardized success response."""
        response = {'data': data}
        if message:
            response['message'] = message
        return response