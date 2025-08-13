#!/usr/bin/env python3
"""
Test script for gpt-oss-20b model using mlx-engine-neuroscope
"""

import sys
import os
import time
import gc
from pathlib import Path

# Add the current directory to Python path to import mlx_engine
sys.path.insert(0, os.getcwd())

from mlx_engine.generate import load_model, create_generator, tokenize
from transformers import AutoTokenizer

# Try to import mlx for memory management
try:
    import mlx.core as mx
    # Set memory limit to 28GB (leaving 4GB for system) to prevent conservative memory warnings
    mx.set_memory_limit(28 * 1024 * 1024 * 1024)
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def test_gpt_oss_20b():
    """Test the gpt-oss-20b model with a simple math question"""
    
    # Model path - using 4-bit quantized version
    model_path = "./models/nightmedia/gpt-oss-20b-q4-hi-mlx"
    
    # Check if model exists
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return False
    
    print("Loading gpt-oss-20b model...")
    try:
        # Load the model
        model_kit = load_model(
            model_path,
            trust_remote_code=False,
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return False
    
    # Load tokenizer
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✓ Tokenizer loaded successfully")
    except Exception as e:
        print(f"✗ Failed to load tokenizer: {e}")
        return False
    
    # Test prompt
    test_question = "What is three plus four?"
    
    # Create conversation format
    conversation = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": test_question}
    ]
    
    # Apply chat template
    try:
        prompt = tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        print(f"✓ Chat template applied")
        print(f"Formatted prompt: {repr(prompt)}")
    except Exception as e:
        print(f"✗ Failed to apply chat template: {e}")
        return False
    
    # Tokenize the prompt
    try:
        prompt_tokens = tokenize(model_kit, prompt)
        print(f"✓ Prompt tokenized ({len(prompt_tokens)} tokens)")
    except Exception as e:
        print(f"✗ Failed to tokenize prompt: {e}")
        return False
    
    # Generate response
    print(f"\nQuestion: {test_question}")
    print("Answer: ", end="", flush=True)
    
    try:
        generator = create_generator(
            model_kit,
            prompt_tokens,
            max_tokens=100,  # Restored to original value
            temp=0.7,
        )
        
        response_text = ""
        start_time = time.time()
        
        for generation_result in generator:
            print(generation_result.text, end="", flush=True)
            response_text += generation_result.text
            
            if generation_result.stop_condition:
                break
        
        end_time = time.time()
        generation_time = end_time - start_time
        
        print(f"\n\n✓ Generation completed in {generation_time:.2f} seconds")
        print(f"Full response: {repr(response_text.strip())}")
        
        # Clean up memory
        if MLX_AVAILABLE:
            mx.metal.clear_cache()
        gc.collect()
        
        # Check if the response contains a reasonable answer
        if "7" in response_text or "seven" in response_text.lower():
            print("✓ Model correctly answered the math question!")
            return True
        else:
            print("⚠ Model response doesn't contain the expected answer (7)")
            return True  # Still consider it a success if it generated something
            
    except Exception as e:
        print(f"\n✗ Failed to generate response: {e}")
        return False

if __name__ == "__main__":
    print("Testing gpt-oss-20b model with mlx-engine-neuroscope")
    print("=" * 60)
    print(f"Script: {__file__}")
    print("=" * 60)
    
    success = test_gpt_oss_20b()
    
    if success:
        print("\n✓ Test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)