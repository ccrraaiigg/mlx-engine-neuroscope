"""MLX Engine API Endpoints Package

This package contains modular endpoint implementations for the MLX Engine API.
Each endpoint category is organized into separate modules for better maintainability.
"""

from .health import register_health_routes
from .models import register_model_routes
from .generation import register_generation_routes
from .activations import register_activation_routes
from .circuits import register_circuit_routes
# Feature routes are registered directly in router.py
from .tasks import register_task_routes

__all__ = [
    'register_health_routes',
    'register_model_routes', 
    'register_generation_routes',
    'register_activation_routes',
    'register_circuit_routes',
    'register_task_routes'
]