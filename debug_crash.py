#!/usr/bin/env python3
"""
Debug the exact cause of the crash
"""

import sys
import os
import threading
import time
import requests
import json

def debug_crash_point():
    """Debug exactly where the crash occurs."""
    print("=== DEBUGGING CRASH POINT ===")
    
    try:
        from mlx_engine.api_server import MLXEngineAPI
        
        # Create API server
        api = MLXEngineAPI()
        
        # Start server in background thread
        def run_server():
            api.run(host='127.0.0.1', port=50114, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        print("1. Testing health...")
        response = requests.get("http://127.0.0.1:50114/health")
        print(f"   Health: {response.status_code}")
        
        print("2. Loading model...")
        model_data = {
            "model_path": "./models/nightmedia/gpt-oss-20b-q4-hi-mlx",
            "model_id": "debug-model"
        }
        response = requests.post("http://127.0.0.1:50114/v1/models/load", json=model_data)
        print(f"   Model load: {response.status_code}")
        
        print("3. Testing basic chat completion (no activations)...")
        chat_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "max_tokens": 5
        }
        response = requests.post("http://127.0.0.1:50114/v1/chat/completions", json=chat_data)
        print(f"   Basic chat: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Basic chat works fine")
        
        print("4. Registering activation hooks...")
        hooks_data = {
            "model": "debug-model",
            "hooks": [{
                "layer_name": "model.layers.0.self_attn",
                "component": "attention",
                "hook_id": "debug_hook",
                "capture_output": True
            }]
        }
        response = requests.post("http://127.0.0.1:50114/v1/activations/hooks", json=hooks_data)
        print(f"   Hooks registration: {response.status_code}")
        
        print("5. Testing chat with activations (THIS IS WHERE IT CRASHES)...")
        print("   About to make the request that causes the crash...")
        
        # This is the request that crashes
        activation_chat_data = {
            "messages": [{"role": "user", "content": "Hi"}],
            "activation_hooks": [{
                "layer_name": "model.layers.0.self_attn",
                "component": "attention", 
                "hook_id": "debug_hook",
                "capture_output": True
            }],
            "max_tokens": 5
        }
        
        print("   Making request to /v1/chat/completions/with_activations...")
        response = requests.post("http://127.0.0.1:50114/v1/chat/completions/with_activations", 
                               json=activation_chat_data)
        print(f"   Activation chat: {response.status_code}")
        
        if response.status_code == 200:
            print("   ✅ SUCCESS - No crash!")
            result = response.json()
            print(f"   Response: {result.get('choices', [{}])[0].get('message', {}).get('content', 'No content')}")
            print(f"   Activations: {len(result.get('activations', {}))}")
        else:
            print(f"   ❌ Request failed: {response.status_code}")
            print(f"   Error: {response.text}")
        
        return True
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run crash debugging."""
    print("Crash Point Debugging")
    print("=" * 40)
    
    success = debug_crash_point()
    
    if success:
        print("\n✅ Debug completed successfully!")
        return 0
    else:
        print("\n❌ Debug failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())