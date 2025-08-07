#!/usr/bin/env python3
"""
Comprehensive demo of the working gpt-oss-20b model with MoE implementation
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
    mx.set_memory_limit(28 * 1024 * 1024 * 1024)
    mx.set_default_device(mx.gpu)
    MLX_AVAILABLE = True
except ImportError:
    MLX_AVAILABLE = False

def demo_gpt_oss_20b():
    """Comprehensive demo of gpt-oss-20b capabilities"""
    
    model_path = "./models/nightmedia/gpt-oss-20b-q4-hi-mlx"
    
    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        return False
    
    print("🚀 Loading gpt-oss-20b model with MoE architecture...")
    try:
        model_kit = load_model(model_path, trust_remote_code=False)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print("✅ Model and tokenizer loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return False
    
    # Test cases showcasing different capabilities
    test_cases = [
        {
            "name": "Math Problem",
            "question": "What is 15 * 23?",
            "max_tokens": 50
        },
        {
            "name": "Creative Writing",
            "question": "Write a short poem about artificial intelligence.",
            "max_tokens": 100
        },
        {
            "name": "Code Generation",
            "question": "Write a Python function to calculate fibonacci numbers.",
            "max_tokens": 150
        },
        {
            "name": "Reasoning",
            "question": "If a train travels 60 mph for 2.5 hours, how far does it go?",
            "max_tokens": 80
        }
    ]
    
    print(f"\n🧪 Running {len(test_cases)} test cases...\n")
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"{'='*60}")
        print(f"Test {i}: {test_case['name']}")
        print(f"{'='*60}")
        
        # Create conversation
        conversation = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": test_case["question"]}
        ]
        
        try:
            # Apply chat template
            prompt = tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # Tokenize
            prompt_tokens = tokenize(model_kit, prompt)
            
            print(f"Question: {test_case['question']}")
            print(f"Tokens: {len(prompt_tokens)}")
            print("Answer: ", end="", flush=True)
            
            # Generate response
            start_time = time.time()
            generator = create_generator(
                model_kit,
                prompt_tokens,
                max_tokens=test_case["max_tokens"],
                temp=0.7,
            )
            
            response_text = ""
            token_count = 0
            
            for generation_result in generator:
                print(generation_result.text, end="", flush=True)
                response_text += generation_result.text
                token_count += 1
                
                if generation_result.stop_condition:
                    break
            
            end_time = time.time()
            generation_time = end_time - start_time
            tokens_per_second = token_count / generation_time if generation_time > 0 else 0
            
            print(f"\n\n📊 Performance:")
            print(f"   Time: {generation_time:.2f}s")
            print(f"   Tokens: {token_count}")
            print(f"   Speed: {tokens_per_second:.1f} tokens/sec")
            
            # Memory cleanup
            if MLX_AVAILABLE:
                mx.clear_cache()
            gc.collect()
            
        except Exception as e:
            print(f"❌ Error in test case {i}: {e}")
            continue
        
        print()
    
    print(f"{'='*60}")
    print("🎉 Demo completed successfully!")
    print(f"{'='*60}")
    
    # Model architecture info
    print("\n🏗️  Model Architecture Summary:")
    print("   • Model: gpt-oss-20b (4-bit quantized)")
    print("   • Architecture: Mixture of Experts (MoE)")
    print("   • Experts: 32 local experts, 4 active per token")
    print("   • Attention: Sliding window + Full attention alternating")
    print("   • Context: 131k tokens max")
    print("   • Quantization: 4-bit with group size 32")
    print("   • RoPE: YARN scaling for long context")
    
    return True

if __name__ == "__main__":
    print("🤖 GPT-OSS-20B Model Demo")
    print("MLX Engine + Neuroscope Integration")
    print("Mixture of Experts Implementation")
    print("=" * 60)
    
    success = demo_gpt_oss_20b()
    
    if success:
        print("\n✅ All tests completed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Demo failed!")
        sys.exit(1)