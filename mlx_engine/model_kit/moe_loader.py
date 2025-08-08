"""
Custom model loader for MoE (Mixture of Experts) models.
"""
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.utils import load_weights, make_shards, get_model_path
from mlx_lm.tokenizer_utils import TokenizerWrapper

from .moe_model import GptOssMoEModel

def load_moe_model(
    model_path: str,
    tokenizer_config: Optional[Dict[str, Any]] = None,
) -> Tuple[nn.Module, TokenizerWrapper]:
    """
    Load a MoE model and its tokenizer from the given path.
    
    Args:
        model_path: Path to the model directory or Hugging Face repo
        tokenizer_config: Optional tokenizer configuration
        
    Returns:
        A tuple containing the model and tokenizer
    """
    # Resolve the model path
    model_path = get_model_path(model_path)
    
    # Load the model configuration
    config_path = model_path / "config.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # Create the model
    model = GptOssMoEModel(config)
    
    # Load the model weights
    weight_files = list(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = list(model_path.glob("*.bin"))
    
    if not weight_files:
        raise ValueError(f"No model weights found in {model_path}")
    
    # Load and apply the weights
    weights = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))
    
    # Load the weights into the model
    model.load_weights(list(weights.items()))
    
    # Load the tokenizer
    try:
        tokenizer = TokenizerWrapper.from_pretrained(model_path)
    except Exception as e:
        raise ValueError(f"Failed to load tokenizer from {model_path}: {e}")
    
    return model, tokenizer

def convert_moe_model(
    model_path: str,
    output_path: str,
    quantize: bool = False,
    q_group_size: int = 64,
    q_bits: int = 4,
    **kwargs
) -> None:
    """
    Convert a MoE model to MLX format.
    
    Args:
        model_path: Path to the source model (Hugging Face repo or local directory)
        output_path: Path to save the converted model
        quantize: Whether to quantize the model
        q_group_size: Group size for quantization
        q_bits: Number of bits for quantization
    """
    from mlx_lm.utils import save_weights, save_config
    
    # Load the model and tokenizer
    model, tokenizer = load_moe_model(model_path)
    
    # Quantize the model if requested
    if quantize:
        from mlx_lm.utils import quantize_model
        model, _ = quantize_model(model, model.config, q_group_size, q_bits)
    
    # Save the model weights
    save_weights(
        str(output_path),
        model,
        save_shards=True,
        **kwargs
    )
    
    # Save the model configuration
    save_config(
        model.config,
        str(Path(output_path) / "config.json")
    )
    
    # Save the tokenizer
    tokenizer.save_pretrained(output_path)
