# Make vision model kit optional
from mlx_engine.logging import log_warn

try:
    from .vision_model_kit import VisionModelKit
    from .vision_model_wrapper import VisionModelWrapper, VLM_AVAILABLE
    from ._transformers_compatibility import (
        fix_qwen2_5_vl_image_processor,
        fix_qwen2_vl_preprocessor,
    )
    
    __all__ = [
        'VisionModelKit',
        'VisionModelWrapper',
        'VLM_AVAILABLE',
        'fix_qwen2_5_vl_image_processor',
        'fix_qwen2_vl_preprocessor',
    ]
    
except ImportError as e:
    log_warn(f"Vision model dependencies not available: {e}")
    log_warn("Vision model features will be disabled.")
    
    # Define dummy classes when dependencies are not available
    class DummyVisionModelKit:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                "Vision model dependencies are not available. "
                "Please install the required packages to use vision model features."
            )
    
    class DummyVisionModelWrapper:
        pass
    
    # Set dummy values for imports
    VisionModelKit = DummyVisionModelKit
    VisionModelWrapper = DummyVisionModelWrapper
    VLM_AVAILABLE = False
    
    # Dummy functions
    def fix_qwen2_5_vl_image_processor(*args, **kwargs):
        raise ImportError("Vision model dependencies are not available.")
    
    def fix_qwen2_vl_preprocessor(*args, **kwargs):
        raise ImportError("Vision model dependencies are not available.")
    
    __all__ = [
        'VisionModelKit',
        'VisionModelWrapper',
        'VLM_AVAILABLE',
        'fix_qwen2_5_vl_image_processor',
        'fix_qwen2_vl_preprocessor',
    ]
