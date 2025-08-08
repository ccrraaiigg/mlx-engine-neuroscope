#!/usr/bin/env python3
"""
Test API routes to debug 404 issue
"""

import sys
import os
import threading
import time
import requests

def test_api_routes():
    """Test that API routes are properly registered."""
    print("Testing API routes...")
    
    try:
        from mlx_engine.api_server import MLXEngineAPI
        
        # Create API server
        api = MLXEngineAPI()
        
        # List all routes
        print("\nRegistered routes:")
        for rule in api.app.url_map.iter_rules():
            methods = ','.join(rule.methods - {'HEAD', 'OPTIONS'})
            print(f"  {methods:10} {rule}")
        
        # Start server in background thread
        def run_server():
            api.run(host='127.0.0.1', port=50112, debug=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        
        # Wait for server to start
        time.sleep(2)
        
        # Test health endpoint
        print("\nTesting endpoints:")
        try:
            response = requests.get("http://127.0.0.1:50112/health")
            print(f"GET /health: {response.status_code}")
        except Exception as e:
            print(f"GET /health: ERROR - {e}")
        
        # Test activation hooks endpoint
        try:
            response = requests.post("http://127.0.0.1:50112/v1/activations/hooks", 
                                   json={"hooks": []})
            print(f"POST /v1/activations/hooks: {response.status_code}")
            if response.status_code != 200:
                print(f"  Response: {response.text}")
        except Exception as e:
            print(f"POST /v1/activations/hooks: ERROR - {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API routes test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run API routes test."""
    print("API Routes Test")
    print("=" * 40)
    
    success = test_api_routes()
    
    if success:
        print("\n✅ API routes test completed")
        return 0
    else:
        print("\n❌ API routes test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())