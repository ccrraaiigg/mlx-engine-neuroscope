#!/usr/bin/env python3
"""
Test activation hooks system
"""

import sys
import os

def test_activation_hook_manager():
    """Test the ActivationHookManager."""
    print("Testing ActivationHookManager...")
    
    try:
        from mlx_engine.activation_hooks_fixed import ActivationHookManager, ActivationHookSpec, ComponentType
        
        # Create a mock model (just an object)
        class MockModel:
            pass
        
        model = MockModel()
        manager = ActivationHookManager(model)
        
        # Test hook registration
        spec = ActivationHookSpec(
            layer_name="model.layers.0.self_attn",
            component=ComponentType.ATTENTION,
            hook_id="test_hook",
            capture_input=False,
            capture_output=True
        )
        
        hook_id = manager.register_hook(spec)
        print(f"✅ Registered hook: {hook_id}")
        
        # Test getting activations
        activations = manager.get_activations()
        print(f"✅ Got activations: {len(activations)} hooks")
        
        # Test clearing activations
        manager.clear_activations()
        print("✅ Cleared activations")
        
        # Test unregistering hook
        manager.unregister_hook(hook_id)
        print("✅ Unregistered hook")
        
        return True
        
    except Exception as e:
        print(f"❌ ActivationHookManager test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_api_server_creation():
    """Test that API server can be created without crashing."""
    print("Testing API server creation...")
    
    try:
        from mlx_engine.api_server import MLXEngineAPI
        
        # Just create the API server (don't run it)
        api = MLXEngineAPI()
        print("✅ API server created successfully")
        
        # Check that the routes are registered
        routes = []
        for rule in api.app.url_map.iter_rules():
            routes.append(str(rule))
        
        activation_routes = [r for r in routes if 'activations' in r]
        print(f"✅ Found {len(activation_routes)} activation routes")
        for route in activation_routes:
            print(f"   - {route}")
        
        return True
        
    except Exception as e:
        print(f"❌ API server creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run activation hooks tests."""
    print("Activation Hooks Test")
    print("=" * 40)
    
    tests = [
        ("ActivationHookManager", test_activation_hook_manager),
        ("API Server Creation", test_api_server_creation),
    ]
    
    results = {}
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name}...")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{'='*40}")
    print("TEST SUMMARY")
    print(f"{'='*40}")
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<30} {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests successful")
    
    if successful_tests == total_tests:
        print("🎉 All activation hooks tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())