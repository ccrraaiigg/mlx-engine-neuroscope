import json
from typing import Callable, Optional, List, Tuple, Dict, Any

import mlx_lm
from mlx_lm.tokenizer_utils import TokenizerWrapper, StreamingDetokenizer

from mlx_engine.cache_wrapper import CacheWrapper
from pathlib import Path
import mlx.nn as nn
import mlx.core as mx
import os

from mlx_engine.logging import log_info, log_warn, log_error
from mlx_engine.model_kit.vision_add_ons.base import BaseVisionAddOn

# Try to import MoE model and loader
try:
    from .moe_model import GptOssMoEModel
    from .moe_loader import load_moe_model
    MOE_AVAILABLE = True
except ImportError as e:
    MOE_AVAILABLE = False
    log_warn(f"MoE model support not available: {e}")
    GptOssMoEModel = None
    load_moe_model = None
try:
    from mlx_engine.model_kit.vision_add_ons.gemma3 import Gemma3VisionAddOn
    GEMMA3_AVAILABLE = True
except ImportError:
    GEMMA3_AVAILABLE = False
    log_warn("Gemma3 vision add-on is not available. Some features may be limited.")

try:
    from mlx_engine.model_kit.vision_add_ons.pixtral import PixtralVisionAddOn
    PIXTRAL_AVAILABLE = True
except ImportError:
    PIXTRAL_AVAILABLE = False
    log_warn("Pixtral vision add-on is not available. Some features may be limited.")

try:
    from mlx_engine.model_kit.vision_add_ons.gemma3n import Gemma3nVisionAddOn
    GEMMA3N_AVAILABLE = True
except ImportError:
    GEMMA3N_AVAILABLE = False
    log_warn("Gemma3n vision add-on is not available. Some features may be limited.")

try:
    from mlx_engine.model_kit.vision_add_ons.mistral3 import Mistral3VisionAddOn
    MISTRAL3_AVAILABLE = True
except ImportError:
    MISTRAL3_AVAILABLE = False
    log_warn("Mistral3 vision add-on is not available. Some features may be limited.")
from mlx_engine.utils.kv_cache_quantization import get_kv_cache_quantization_params
from mlx_engine.utils.prompt_processing import process_prompt_text_only
from mlx_engine.activation_hooks import ActivationHookManager, ActivationHookSpec, ComponentType

LOG_PREFIX = "ModelKit"


class ModelKit:
    """
    Collection of objects and methods that are needed for operating a model.

    Args:
        model_path (Path): Path to the model directory containing model files.
        vocab_only (bool): Only load vocabulary/tokenizer, not the full model.
        max_kv_size (int): Maximum size of the key-value cache used during model inference.
        kv_bits (Optional[int]): Number of bits for KV cache quantization. None disables quantization.
        kv_group_size (Optional[int]): Group size for KV cache quantization. Defaults to 64.
        quantized_kv_start (Optional[int]): Start index for KV cache quantization. Defaults to 0.
    """

    # Initialize VISION_ADD_ON_MAP with only the available vision add-ons
    VISION_ADD_ON_MAP = {}
    
    # Add Gemma3 if available
    if GEMMA3_AVAILABLE:
        VISION_ADD_ON_MAP["gemma3"] = Gemma3VisionAddOn
    
    # Add Gemma3n if available
    if GEMMA3N_AVAILABLE:
        VISION_ADD_ON_MAP["gemma3n"] = Gemma3nVisionAddOn
    
    # Add Pixtral if available
    if PIXTRAL_AVAILABLE:
        VISION_ADD_ON_MAP["pixtral"] = PixtralVisionAddOn
    
    # Add Mistral3 if available
    if MISTRAL3_AVAILABLE:
        VISION_ADD_ON_MAP["mistral3"] = Mistral3VisionAddOn

    # model state tracking
    model: nn.Module = None
    tokenizer: TokenizerWrapper = None
    detokenizer: StreamingDetokenizer = None
    cache_wrapper: Optional[CacheWrapper] = None
    _cross_prompt_cache_active: bool = False
    max_kv_size: Optional[int] = None
    kv_bits: Optional[int] = None
    kv_group_size: Optional[int] = None
    quantized_kv_start: Optional[int] = None
    draft_model: Optional[nn.Module] = None
    model_type: Optional[str] = None

    # multi-modal add-ons
    vision_add_on: Optional[BaseVisionAddOn] = None
    
    # activation hooks for interpretability
    activation_hook_manager: Optional[ActivationHookManager] = None

    def _vocab_only_init(self, model_path: Path):
        log_info(
            prefix=LOG_PREFIX,
            message=f"Loading model (vocab-only) from {model_path}...",
        )
        self.tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(model_path)
        self.detokenizer = self.tokenizer.detokenizer
        log_info(prefix=LOG_PREFIX, message="Model (vocab-only) loaded successfully")

    def _load_model_with_retry(
        self,
        model_path: Path,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ) -> Tuple[nn.Module, TokenizerWrapper]:
        """
        Try to load the model with standard MLX-LM, falling back to custom MoE loader if needed.
        
        Args:
            model_path: Path to the model directory
            max_kv_size: Maximum size of the KV cache
            kv_bits: Number of bits for KV cache quantization
            kv_group_size: Group size for KV cache quantization
            quantized_kv_start: Start index for KV cache quantization
            
        Returns:
            Tuple of (model, tokenizer)
        """
        try:
            # First try standard MLX-LM loading
            model, tokenizer = mlx_lm.utils.load(model_path)
            log_info(prefix=LOG_PREFIX, message="Model loaded with standard MLX-LM loader")
            return model, tokenizer
        except (ValueError, RuntimeError, AttributeError) as e:
            # Check if this is an MoE model
            config_path = model_path / "config.json"
            if not config_path.exists():
                raise ValueError(f"Config file not found at {config_path}")
                
            with open(config_path, "r") as f:
                config = json.load(f)
                
            is_moe = any(key in config for key in ["num_local_experts", "num_experts_per_tok"])
            
            if is_moe and MOE_AVAILABLE:
                log_info(prefix=LOG_PREFIX, message="Detected MoE model, using custom loader...")
                try:
                    model, tokenizer = load_moe_model(str(model_path))
                    log_info(prefix=LOG_PREFIX, message="MoE model loaded successfully")
                    return model, tokenizer
                except Exception as moe_error:
                    log_error(prefix=LOG_PREFIX, message=f"Failed to load MoE model: {moe_error}")
                    raise
            
            # Re-raise the original error if we can't handle it
            log_error(prefix=LOG_PREFIX, message=f"Failed to load model: {e}")
            raise

    def _full_model_init(
        self,
        model_path: Path,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        kv_bits, kv_group_size, quantized_kv_start = get_kv_cache_quantization_params(
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )
        if kv_bits and max_kv_size is not None:
            # Quantized KV cache is only supported for non-rotating KV cache
            log_warn(
                prefix=LOG_PREFIX,
                message="max_kv_size is ignored when using KV cache quantization",
            )
            max_kv_size = None
            
        self.model_path = model_path
        log_info(prefix=LOG_PREFIX, message=f"Loading model from {model_path}...")
        
        # Load the config to determine model type
        config_path = model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")
            
        with open(config_path, "r") as f:
            config = json.load(f)
            
        self.model_type = config.get("model_type", None)
        self.is_moe = any(key in config for key in ["num_local_experts", "num_experts_per_tok"])
        
        # Load the model and tokenizer
        self.model, self.tokenizer = self._load_model_with_retry(
            model_path,
            max_kv_size,
            kv_bits,
            kv_group_size,
            quantized_kv_start,
        )
        
        # Initialize the cache wrapper
        self.cache_wrapper = CacheWrapper(
            self.model,
            max_kv_size,
            kv_bits=kv_bits,
            kv_group_size=kv_group_size,
            quantized_kv_start=quantized_kv_start,
        )
        self.kv_bits = kv_bits
        self.kv_group_size = kv_group_size
        self.quantized_kv_start = quantized_kv_start
        vision_add_on_class = self.VISION_ADD_ON_MAP.get(self.model_type)
        should_load_vision_add_on = (
            vision_add_on_class is not None and "vision_config" in config_json
        )
        if should_load_vision_add_on:
            self.vision_add_on = vision_add_on_class(model_path)
        
        # Initialize activation hook manager
        self.activation_hook_manager = ActivationHookManager(self.model)
        
        log_info(prefix=LOG_PREFIX, message="Model loaded successfully")

    def __init__(
        self,
        model_path: Path,
        vocab_only: bool = False,
        max_kv_size: Optional[int] = None,
        kv_bits: Optional[int] = None,
        kv_group_size: Optional[int] = None,
        quantized_kv_start: Optional[int] = None,
    ):
        if vocab_only:
            self._vocab_only_init(model_path)
        else:
            self._full_model_init(
                model_path,
                max_kv_size,
                kv_bits,
                kv_group_size,
                quantized_kv_start,
            )

    def tokenize(self, prompt: str) -> List[int]:
        ids = self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(prompt))
        if isinstance(ids, int):
            return [ids]
        return ids

    def process_prompt(
        self,
        prompt_tokens,
        images_b64: Optional[List[str]],
        prompt_progress_callback: Optional[Callable[[float], bool]],
        generate_args: dict,
        speculative_decoding_toggle: Optional[bool] = None,
    ) -> Tuple[mx.array, Optional[mx.array]]:
        ### TEXT-ONLY PROCESS_PROMPT ###
        is_text_only_processing = images_b64 is None or len(images_b64) == 0
        if is_text_only_processing:
            self._cross_prompt_cache_active = True
            if len(prompt_tokens) == 0:
                log_warn(
                    prefix="ModelKit",
                    message="Received empty prompt. Generation quality will likely be poor",
                )
                # Models expect some sort of input, so add whitespace
                prompt_tokens = self.tokenize(" ")
            return process_prompt_text_only(
                mx.array(prompt_tokens),
                self.cache_wrapper,
                generate_args,
                self.draft_model,
                speculative_decoding_toggle,
                prompt_progress_callback,
            ), None
        ### WITH IMAGES PROMPT PROCESSING ###s
        if self.vision_add_on is None:
            raise ValueError(
                "Vision add-on is not loaded, but images were provided for processing"
            )
        self._cross_prompt_cache_active = False
        input_ids, embeddings = self.vision_add_on.compute_embeddings(
            self.model, prompt_tokens, images_b64
        )
        return input_ids, embeddings

    def is_cross_prompt_cache_active(self) -> bool:
        """
        Check if cross-prompt caching is currently enabled.
        Can be overridden by subclasses for custom behavior.
        """
        return self._cross_prompt_cache_active

    def record_token_to_cache(self, token: int) -> None:
        self.cache_wrapper.record_generated_token(token)

    @staticmethod
    def is_supported_vision_arch(model_arch: str) -> bool:
        """
        Determines if the specified model architecture has vision support.

        Args:
            model_arch (str): The model architecture identifier to check

        Returns:
            bool: True if vision is supported, False otherwise
        """
        return model_arch in ModelKit.VISION_ADD_ON_MAP

    def is_draft_model_compatible(self, path: 'str | Path') -> bool:
        path = Path(path)
        if self.tokenizer is None:
            log_warn(
                prefix=LOG_PREFIX,
                message="Draft model compatibility check requires at least a vocab-only "
                "loaded main model",
            )
            return False
        if self.vision_add_on is not None:
            log_warn(
                prefix=LOG_PREFIX,
                message="Draft models are currently unsupported for vision models",
            )
            return False
        draft_tokenizer = mlx_lm.tokenizer_utils.load_tokenizer(path)
        if draft_tokenizer.vocab_size != self.tokenizer.vocab_size:
            return False
        return True

    def load_draft_model(self, path: 'str | Path') -> None:
        log_info(prefix=LOG_PREFIX, message=f"Loading draft model from {path}...")
        path = Path(path)
        if self.model is None:
            raise ValueError("Main model must be loaded before loading a draft model")
        if not self.is_draft_model_compatible(path):
            raise ValueError("Draft model is not compatible with main model")
        self.draft_model, _ = mlx_lm.utils.load(path)
        self.cache_wrapper.set_draft_model(self.draft_model)
        log_info(prefix=LOG_PREFIX, message="Draft model loaded")

    def unload_draft_model(self) -> None:
        if self.draft_model is None:
            log_info(prefix=LOG_PREFIX, message="No loaded draft model to unload")
        else:
            self.draft_model = None
            self.cache_wrapper.unset_draft_model()
        # Noticed that draft model memory would not be released without clearing metal cache
        mx.clear_cache()

    def register_activation_hook(self, layer_name: str, component: str, 
                                hook_id: Optional[str] = None,
                                capture_input: bool = False,
                                capture_output: bool = True) -> str:
        """Register an activation hook for interpretability analysis."""
        if self.activation_hook_manager is None:
            raise ValueError("Activation hooks not available - model not fully loaded")
        
        # Convert string component to enum
        try:
            component_type = ComponentType(component.lower())
        except ValueError:
            raise ValueError(f"Unknown component type: {component}")
        
        spec = ActivationHookSpec(
            layer_name=layer_name,
            component=component_type,
            hook_id=hook_id,
            capture_input=capture_input,
            capture_output=capture_output
        )
        
        return self.activation_hook_manager.register_hook(spec)
    
    def unregister_activation_hook(self, hook_id: str):
        """Unregister an activation hook."""
        if self.activation_hook_manager is not None:
            self.activation_hook_manager.unregister_hook(hook_id)
    
    def clear_activation_hooks(self):
        """Clear all activation hooks."""
        if self.activation_hook_manager is not None:
            self.activation_hook_manager.clear_all_hooks()
    
    def get_captured_activations(self, hook_id: Optional[str] = None, 
                                clear_after_get: bool = True) -> dict:
        """Get captured activations from hooks."""
        print(f"\n[DEBUG] ModelKit.get_captured_activations called")
        print(f"[DEBUG]   hook_id: {hook_id}")
        print(f"[DEBUG]   clear_after_get: {clear_after_get}")
        
        if self.activation_hook_manager is None:
            print("[WARNING] No activation hook manager, returning empty dict")
            return {}
        
        try:
            print("[DEBUG] Getting activations from hook manager...")
            activations = self.activation_hook_manager.get_activations(hook_id)
            print(f"[DEBUG] Retrieved activations with keys: {list(activations.keys())}")
            
            if clear_after_get:
                print("[DEBUG] Clearing activations after get")
                self.activation_hook_manager.clear_activations(hook_id)
            
            return activations
        except Exception as e:
            print(f"[ERROR] Failed to get activations: {e}")
            return {}
    
    def clear_captured_activations(self, hook_id: Optional[str] = None):
        """Clear captured activations without getting them."""
        print(f"\n[DEBUG] ModelKit.clear_captured_activations called")
        print(f"[DEBUG]   hook_id: {hook_id}")
        
        if self.activation_hook_manager is not None:
            print(f"[DEBUG] Clearing activations for hook: {hook_id}")
            try:
                self.activation_hook_manager.clear_activations(hook_id)
                print("[DEBUG] Successfully cleared activations")
            except Exception as e:
                print(f"[ERROR] Failed to clear activations: {e}")
                raise
        else:
            print("[WARNING] No activation hook manager to clear activations from")
