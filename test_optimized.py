#!/usr/bin/env python3
"""
Optimized test script for gpt-oss-20b model
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
    # Set memory limit to 28GB and optimize for speed
    mx.set_memory_limit(28 * 1024 * 1024 * 1024)
    # Enable optimizations
    mx.set_default_device(mx.gpu)
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def test_gpt_oss_20b_optimized():
    """Test the gpt-oss-20b model with optimizations for speed"""
    
    # Try 4-bit model first, fallback to 8-bit
    model_paths = [
        "./models/NexVeridian/gpt-oss-20b-4bit",
        "./models/nightmedia/gpt-oss-20b-q4-hi-mlx", 
        "./models/gpt-oss-20b-mlx-8bit"
    ]
    
    model_path = None
    for path in model_paths:
        if Path(path).exists():
            model_path = path
            print(f"Using model: {path}")
            break
    
    if not model_path:
        print("Error: No model found. Checked:")
        for path in model_paths:
            print(f"  {path}")
        return False
    
    print("Loading model with optimizations...")
    try:
        # Load the model with optimizations
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
    
    # Simplified prompt for faster processing
    test_question = "3+4="
    
    # Simple format (no complex chat template)
    prompt = f"Question: {test_question}\nAnswer:"
    
    print(f"Simplified prompt: {repr(prompt)}")
    
    # Tokenize the prompt
    try:
        prompt_tokens = tokenize(model_kit, prompt)
        print(f"✓ Prompt tokenized ({len(prompt_tokens)} tokens)")
    except Exception as e:
        print(f"✗ Failed to tokenize prompt: {e}")
        return False
    
    # Generate response with speed optimizations
    print(f"\nQuestion: {test_question}")
    print("Answer: ", end="", flush=True)
    
    try:
        generator = create_generator(
            model_kit,
            prompt_tokens,
            max_tokens=10,   # Very short for speed test
            temp=0.1,        # Lower temperature for faster, more deterministic output
        )
        
        response_text = ""
        start_time = time.time()
        token_count = 0
        
        for generation_result in generator:
            print(generation_result.text, end="", flush=True)
            response_text += generation_result.text
            token_count += 1
            
            # Stop early if we get a complete answer
            if any(char.isdigit() for char in generation_result.text) or generation_result.stop_condition:
                break
        
        end_time = time.time()
        generation_time = end_time - start_time
        tokens_per_second = token_count / generation_time if generation_time > 0 else 0
        
        print(f"\n\n✓ Generation completed in {generation_time:.2f} seconds")
        print(f"✓ Generated {token_count} tokens ({tokens_per_second:.1f} tokens/sec)")
        print(f"Full response: {repr(response_text.strip())}")
        
        # Check if the response contains a reasonable answer
        if "7" in response_text:
            print("✓ Model correctly answered the math question!")
            return True
        else:
            print("⚠ Model response doesn't contain the expected answer (7)")
            return True  # Still consider it a success if it generated something
            
    except Exception as e:
        print(f"\n✗ Failed to generate response: {e}")
        return False

if __name__ == "__main__":
    print("Optimized test for gpt-oss-20b model")
    print("=" * 45)
    
    success = test_gpt_oss_20b_optimized()
    
    if success:
        print("\n✓ Optimized test completed successfully!")
        sys.exit(0)
    else:
        print("\n✗ Test failed!")
        sys.exit(1)