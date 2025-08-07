#!/usr/bin/env python3
"""
NeuroScope Demo Runner

Simple script to run the NeuroScope REST interface demo with proper setup.
This elaborates on test_gpt_oss_20b.py by showing the full REST workflow.
"""

import sys
import os
import subprocess
import time
import signal
from pathlib import Path

def check_requirements():
    """Check if required dependencies are available."""
    print("Checking requirements...")
    
    # Check if model exists
    model_path = Path("./models/nightmedia/gpt-oss-20b-q4-hi-mlx")
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print("Please ensure the model is downloaded and available.")
        return False
    
    print(f"✅ Model found at {model_path}")
    
    # Check Python dependencies
    try:
        import flask
        import mlx.core
        print("✅ Required dependencies available")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install required packages:")
        print("pip install flask flask-cors mlx")
        return False

def run_basic_test():
    """Run the basic test from test_gpt_oss_20b.py for comparison."""
    print("\n" + "="*60)
    print("RUNNING BASIC TEST (from test_gpt_oss_20b.py)")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "test_gpt_oss_20b.py"
        ], capture_output=True, text=True, timeout=120)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ Basic test passed!")
            return True
        else:
            print("❌ Basic test failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ Basic test timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running basic test: {e}")
        return False

def run_neuroscope_demo():
    """Run the NeuroScope REST interface demo."""
    print("\n" + "="*60)
    print("RUNNING NEUROSCOPE REST INTERFACE DEMO")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "demo_neuroscope_rest_interface.py"
        ], capture_output=True, text=True, timeout=300)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print("✅ NeuroScope demo completed!")
            return True
        else:
            print("❌ NeuroScope demo failed!")
            return False
            
    except subprocess.TimeoutExpired:
        print("❌ NeuroScope demo timed out!")
        return False
    except Exception as e:
        print(f"❌ Error running NeuroScope demo: {e}")
        return False

def show_api_reference():
    """Show the API reference."""
    print("\n" + "="*60)
    print("NEUROSCOPE API REFERENCE")
    print("="*60)
    
    try:
        result = subprocess.run([
            sys.executable, "neuroscope_api_reference.py"
        ], capture_output=True, text=True, timeout=30)
        
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error showing API reference: {e}")
        return False

def main():
    """Main demo runner."""
    print("NeuroScope MLX Engine Demo Runner")
    print("=" * 50)
    print("This demo elaborates on test_gpt_oss_20b.py by showing")
    print("how NeuroScope will interact with the MLX Engine REST API.")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Requirements check failed!")
        print("Please install dependencies and ensure model is available.")
        return 1
    
    # Run tests in sequence
    tests = [
        ("Basic Test", run_basic_test),
        ("NeuroScope Demo", run_neuroscope_demo),
        ("API Reference", show_api_reference)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n🚀 Starting {test_name}...")
        try:
            results[test_name] = test_func()
        except KeyboardInterrupt:
            print(f"\n⚠️  {test_name} interrupted by user")
            results[test_name] = False
            break
        except Exception as e:
            print(f"\n❌ {test_name} failed with exception: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("DEMO SUMMARY")
    print("="*60)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    successful_tests = sum(results.values())
    total_tests = len(results)
    
    print(f"\nOverall: {successful_tests}/{total_tests} tests successful")
    
    if successful_tests == total_tests:
        print("\n🎉 All demos completed successfully!")
        print("\nThe NeuroScope REST interface is ready for integration.")
        print("\nKey features demonstrated:")
        print("- ✅ Basic model loading and generation")
        print("- ✅ REST API server functionality") 
        print("- ✅ Activation capture during generation")
        print("- ✅ Multiple analysis scenarios")
        print("- ✅ Comprehensive API reference")
        
        print("\nNext steps:")
        print("1. Implement the REST client in NeuroScope")
        print("2. Add activation visualization components")
        print("3. Create circuit analysis workflows")
        print("4. Test with real mechanistic interpretability tasks")
        
        return 0
    else:
        print("\n⚠️  Some demos failed.")
        print("Check the output above for details.")
        print("\nCommon issues:")
        print("- Model not found (download required)")
        print("- Missing dependencies (pip install flask mlx)")
        print("- Memory limitations (reduce model size)")
        
        return 1

if __name__ == "__main__":
    try:
        exit_code = main()
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Demo failed with unexpected error: {e}")
        sys.exit(1)