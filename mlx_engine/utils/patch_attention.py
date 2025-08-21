"""Patch MLX scaled_dot_product_attention to handle dtype promotion issues."""

import mlx.core as mx
from typing import Optional
import sys

# Store the original function before we patch anything
_original_scaled_dot_product_attention = None

def patched_scaled_dot_product_attention(
    q: mx.array,
    k: mx.array, 
    v: mx.array,
    *,
    scale: Optional[float] = None,
    mask: Optional[mx.array] = None,
    stream: Optional[mx.Stream] = None
) -> mx.array:
    """
    Patched version of scaled_dot_product_attention that handles dtype promotion.
    
    This wrapper ensures that the mask dtype is compatible with the query/key/value dtypes
    to prevent '[scaled_dot_product_attention] Mask type must promote to output type' errors.
    
    Args:
        q: Query tensor with shape [B, N_q, T_q, D]
        k: Key tensor with shape [B, N_kv, T_kv, D] 
        v: Value tensor with shape [B, N_kv, T_kv, D]
        scale: Scale for queries (typically 1.0 / sqrt(q.shape[-1]))
        mask: Attention mask (boolean or additive)
        stream: MLX stream for computation
    """
    # If mask is provided, ensure it has the correct dtype
    if mask is not None and not isinstance(mask, str):
        # Get the promoted dtype of queries, keys, values
        target_dtype = q.dtype
        
        # Cast mask to match the target dtype if needed
        if mask.dtype != target_dtype:
            mask = mask.astype(target_dtype)
    
    # Call the original function with the corrected mask
    kwargs = {}
    if scale is not None:
        kwargs['scale'] = scale
    if mask is not None:
        kwargs['mask'] = mask
    if stream is not None:
        kwargs['stream'] = stream
        
    return _original_scaled_dot_product_attention(q, k, v, **kwargs)

def patch_mlx_attention():
    """
    Patch the MLX scaled_dot_product_attention function to handle dtype promotion.
    
    This function replaces the original scaled_dot_product_attention with our
    patched version that automatically handles mask dtype casting.
    """
    global _original_scaled_dot_product_attention
    
    # Only patch if we haven't already
    if _original_scaled_dot_product_attention is None:
        try:
            # Import and store the original function from mlx.core.fast
            import mlx.core.fast as mx_fast
            _original_scaled_dot_product_attention = mx_fast.scaled_dot_product_attention
            
            # Replace with our patched version
            mx_fast.scaled_dot_product_attention = patched_scaled_dot_product_attention
            
            # Also patch the module in sys.modules to ensure any other imports get our version
            if 'mlx.core.fast' in sys.modules:
                sys.modules['mlx.core.fast'].scaled_dot_product_attention = patched_scaled_dot_product_attention
                
        except (ImportError, AttributeError):
            # MLX fast module not available or function doesn't exist, skip patching
            pass

def unpatch_mlx_attention():
    """
    Restore the original MLX scaled_dot_product_attention function.
    """
    global _original_scaled_dot_product_attention
    
    if _original_scaled_dot_product_attention is not None:
        try:
            # Restore the original function
            import mlx.core.fast as mx_fast
            mx_fast.scaled_dot_product_attention = _original_scaled_dot_product_attention
            
            if 'mlx.core.fast' in sys.modules:
                sys.modules['mlx.core.fast'].scaled_dot_product_attention = _original_scaled_dot_product_attention
                
        except (ImportError, AttributeError):
            pass
        
        # Reset the global variable
        _original_scaled_dot_product_attention = None
        
        _original_scaled_dot_product_attention = None