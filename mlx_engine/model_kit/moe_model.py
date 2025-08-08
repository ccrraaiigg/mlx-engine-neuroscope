"""
Implementation of MoE (Mixture of Experts) model for gpt-oss-20b.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import mlx.core as mx
import mlx.nn as nn
import math

@dataclass
class MoEConfig:
    """Configuration for MoE (Mixture of Experts) layers."""
    num_experts: int
    num_experts_per_tok: int
    hidden_size: int
    intermediate_size: int
    router_aux_loss_coef: float = 0.1
    dtype: str = "float32"

class MoELayer(nn.Module):
    """A Mixture of Experts layer with a router and multiple expert networks."""
    
    def __init__(self, config: MoEConfig):
        super().__init__()
        self.config = config
        
        # Router that decides which experts to use for each token
        self.router = nn.Linear(
            config.hidden_size, 
            config.num_experts,
            bias=False
        )
        
        # Create multiple expert networks
        self.experts = [
            nn.Sequential(
                nn.Linear(config.hidden_size, config.intermediate_size, bias=False),
                nn.GELU(),
                nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
            )
            for _ in range(config.num_experts)
        ]
        
        self.aux_loss = 0.0
    
    def __call__(self, x: mx.array) -> mx.array:
        """Forward pass through the MoE layer."""
        batch_size, seq_len, hidden_dim = x.shape
        x_flat = x.reshape(-1, hidden_dim)  # Combine batch and sequence dimensions
        
        # Get router logits and probabilities
        router_logits = self.router(x_flat)
        router_probs = mx.softmax(router_logits, axis=-1)
        
        # Select top-k experts for each token
        topk_probs, topk_indices = mx.topk(
            router_probs, 
            k=self.config.num_experts_per_tok,
            axis=-1
        )
        
        # Normalize probabilities
        topk_probs = topk_probs / topk_probs.sum(axis=-1, keepdims=True)
        
        # Initialize output
        output = mx.zeros_like(x_flat)
        
        # Process through selected experts
        for expert_idx, expert in enumerate(self.experts):
            # Find which tokens are assigned to this expert
            expert_mask = (topk_indices == expert_idx).any(axis=-1)
            if not expert_mask.any():
                continue
                
            # Process tokens with this expert
            expert_input = x_flat[expert_mask]
            expert_output = expert(expert_input)
            
            # Get the weights for this expert's outputs
            expert_weights = topk_probs[expert_mask, :] * (topk_indices[expert_mask, :] == expert_idx)
            expert_weights = expert_weights.sum(axis=-1, keepdims=True)
            
            # Add weighted expert output to the result
            output = output.at[expert_mask].add(expert_output * expert_weights)
        
        # Reshape back to original dimensions
        output = output.reshape(batch_size, seq_len, hidden_dim)
        
        # Calculate auxiliary loss for balanced expert usage
        self.aux_loss = self._compute_aux_loss(router_probs, topk_indices)
        
        return output
    
    def _compute_aux_loss(self, router_probs: mx.array, expert_indices: mx.array) -> float:
        """Compute the auxiliary loss for balanced expert usage."""
        # Calculate the fraction of tokens assigned to each expert
        expert_mask = mx.eye(self.config.num_experts)[expert_indices]  # [batch*seq, k, num_experts]
        expert_mask = expert_mask.sum(axis=1)  # [batch*seq, num_experts]
        
        # Calculate mean probability of selecting each expert
        mean_probs = router_probs.mean(axis=0)  # [num_experts]
        
        # Calculate fraction of tokens assigned to each expert
        expert_fraction = expert_mask.mean(axis=0)  # [num_experts]
        
        # Auxiliary loss is the dot product of mean_probs and expert_fraction
        aux_loss = (mean_probs * expert_fraction).sum()
        
        # Scale by the coefficient and return
        return self.config.router_aux_loss_coef * aux_loss

class GptOssMoEModel(nn.Module):
    """GPT-Oss model with MoE (Mixture of Experts) support."""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Model dimensions
        self.hidden_size = config["hidden_size"]
        self.num_attention_heads = config["num_attention_heads"]
        self.num_key_value_heads = config.get("num_key_value_heads", self.num_attention_heads)
        self.head_dim = config.get("head_dim", self.hidden_size // self.num_attention_heads)
        self.num_hidden_layers = config["num_hidden_layers"]
        self.rms_norm_eps = config.get("rms_norm_eps", 1e-5)
        
        # MoE configuration
        self.num_experts = config.get("num_local_experts", 32)
        self.num_experts_per_tok = config.get("num_experts_per_tok", 4)
        self.intermediate_size = config.get("intermediate_size", 4 * self.hidden_size)
        
        # Token embeddings
        self.embed_tokens = nn.Embedding(config["vocab_size"], self.hidden_size)
        
        # Attention layers
        self.layers = [
            self._create_decoder_layer(layer_idx)
            for layer_idx in range(self.num_hidden_layers)
        ]
        
        # Final layer norm
        self.norm = nn.RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(self.hidden_size, config["vocab_size"], bias=False)
    
    def _create_decoder_layer(self, layer_idx: int) -> nn.Module:
        """Create a single decoder layer with MoE."""
        # Create MoE config
        moe_config = MoEConfig(
            num_experts=self.num_experts,
            num_experts_per_tok=self.num_experts_per_tok,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            router_aux_loss_coef=self.config.get("router_aux_loss_coef", 0.1)
        )
        
        # Create a single decoder layer with MoE
        return nn.TransformerDecoderLayer(
            dims=self.hidden_size,
            num_heads=self.num_attention_heads,
            num_heads_kv=self.num_key_value_heads,
            mlp_dims=self.intermediate_size,
            norm_eps=self.rms_norm_eps,
            custom_moe_layer=MoELayer(moe_config)
        )
    
    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
        **kwargs
    ) -> Tuple[mx.array, Optional[float]]:
        """Forward pass through the model."""
        # Get input embeddings
        inputs_embeds = self.embed_tokens(input_ids)
        
        # Run through decoder layers
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states,
                attention_mask=attention_mask,
                **kwargs
            )
        
        # Apply final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Get logits
        logits = self.lm_head(hidden_states)
        
        # Calculate total auxiliary loss
        aux_loss = sum(layer.self_attn.aux_loss for layer in self.layers)
        
        return logits, aux_loss
