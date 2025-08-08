#!/usr/bin/env python3
"""
Test the activation hooks endpoint directly
"""

import sys
import os
import threading
import time
import requests
import json

def test_activation_endpoint():
    """Test the activation hooks endpoint directly."""
    print("Testing activation hooks endpoint...")
    
    try:
        from mlx_engine.api_server import MLXEngineAPI
        from mlx_engine import load_model
        
        # Create API server
        api = MLXEngineAPI()
        
        # Start server in background thread
        def run_server():
            api.run(host='127.0.0.1', port=50113, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(3)
        
        print("1. Testing health endpoint...")
        try:
            response = requests.get("http://127.0.0.1:50113/health")
            print(f"   Health: {response.status_code} - {response.json()}")
        except Exception as e:
            print(f"   Health failed: {e}")
            return False
        
        print("2. Loading model...")
        try:
            model_data = {
                "model_path": "./models/nightmedia/gpt-oss-20b-q4-hi-mlx",
                "model_id": "test-model"
            }
            response = requests.post("http://127.0.0.1:50113/v1/models/load", json=model_data)
            print(f"   Model load: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Model ID: {result.get('model_id')}")
                print(f"   Supports activations: {result.get('supports_activations')}")
            else:
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"   Model load failed: {e}")
            return False
        
        print("3. Testing activation hooks endpoint...")
        try:
            hooks_data = {
                "model": "test-model",
                "hooks": [
                    {
                        "layer_name": "model.layers.0.self_attn",
                        "component": "attention",
                        "hook_id": "test_hook",
                        "capture_output": True
                    }
                ]
            }
            response = requests.post("http://127.0.0.1:50113/v1/activations/hooks", json=hooks_data)
            print(f"   Hooks registration: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"   Registered hooks: {result.get('registered_hooks', [])}")
                return True
            else:
                print(f"   Error: {response.text}")
                return False
        except Exception as e:
            print(f"   Hooks registration failed: {e}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run activation endpoint test."""
    print("Activation Endpoint Test")
    print("=" * 40)
    
    success = test_activation_endpoint()
    
    if success:
        print("\n✅ Activation endpoint test passed!")
        return 0
    else:
        print("\n❌ Activation endpoint test failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())