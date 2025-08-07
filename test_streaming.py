#!/usr/bin/env python3
"""
Streaming test with aggressive memory management
"""

import sys
import os
import gc
import time
from pathlib import Path

sys.path.insert(0, os.getcwd())

def test_streaming():
    """Test with streaming and memory management"""
    
    model_path = "./models/nightmedia/gpt-oss-20b-q4-hi-mlx"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return False
    
    print("Attempting streaming generation...")
    
    try:
        # Import only when needed
        from mlx_engine.generate import load_model, create_generator, tokenize
        from transformers import AutoTokenizer
        
        # Try to import mlx for memory management
        try:
            import mlx.core as mx
            MLX_AVAILABLE = True
        except ImportError:
            MLX_AVAILABLE = False
        
        print("Loading model (this will use significant memory)...")
        
        # Load with minimal settings
        model_kit = load_model(model_path, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print("✓ Model loaded")
        
        # Very simple prompt
        prompt = "3+4="
        prompt_tokens = tokenize(model_kit, prompt)
        
        print(f"Generating with {len(prompt_tokens)} input tokens...")
        
        # Create generator with minimal settings
        generator = create_generator(
            model_kit,
            prompt_tokens,
            max_tokens=3,    # Extremely short
            temp=0.0,        # Deterministic
        )
        
        # Stream tokens one by one with cleanup
        response = ""
        token_count = 0
        
        for result in generator:
            response += result.text
            token_count += 1
            print(f"Token {token_count}: '{result.text}'")
            
            # Aggressive memory cleanup after each token
            if MLX_AVAILABLE and token_count % 1 == 0:
                mx.clear_cache()
            gc.collect()
            
            # Hard stop to prevent memory issues
            if token_count >= 2 or result.stop_condition:
                break
        
        print(f"✓ Final response: '{response.strip()}'")
        
        # Final cleanup
        del generator, model_kit, tokenizer
        if MLX_AVAILABLE:
            mx.clear_cache()
        gc.collect()
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        # Emergency cleanup
        try:
            if 'mx' in locals() and MLX_AVAILABLE:
                mx.clear_cache()
        except:
            pass
        gc.collect()
        return False

if __name__ == "__main__":
    print("Streaming test with memory management")
    print("=" * 40)
    
    success = test_streaming()
    
    if success:
        print("\n✓ Streaming test completed!")
    else:
        print("\n✗ Streaming test failed!")
        print("\nRecommendation: Use a smaller model or increase system memory.")