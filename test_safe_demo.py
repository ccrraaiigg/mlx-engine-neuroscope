#!/usr/bin/env python3
"""
Safe test of the demo fixes
"""

import sys
import os

def test_basic_imports():
    """Test that basic imports work without crashing."""
    print("Testing basic imports...")
    
    try:
        from mlx_engine.activation_hooks_fixed import ActivationHookSpec, ComponentType
        print("✅ Activation hooks imports work")
    except Exception as e:
        print(f"❌ Activation hooks import failed: {e}")
        return False
    
    try:
        from mlx_engine.api_server import MLXEngineAPI
        print("✅ API server imports work")
    except Exception as e:
        print(f"❌ API server import failed: {e}")
        return False
    
    return True

def test_activation_hook_spec():
    """Test that ActivationHookSpec can be created safely."""
    print("Testing ActivationHookSpec creation...")
    
    try:
        from mlx_engine.activation_hooks_fixed import ActivationHookSpec, ComponentType
        
        spec = ActivationHookSpec(
            layer_name="model.layers.0.self_attn",
            component=ComponentType.ATTENTION,
            hook_id="test_hook",
            capture_input=False,
            capture_output=True
        )
        print(f"✅ Created hook spec: {spec.hook_id}")
        return True
    except Exception as e:
        print(f"❌ Hook spec creation failed: {e}")
        return False

def main():
    """Run safe tests."""
    print("Safe Demo Test")
    print("=" * 40)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Hook Spec Creation", test_activation_hook_spec),
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
        print("🎉 All safe tests passed!")
        return 0
    else:
        print("⚠️  Some tests failed.")
        return 1

if __name__ == "__main__":
    sys.exit(main())