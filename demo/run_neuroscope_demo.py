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
import glob

def get_next_run_number():
    """Get the next available run number for logs and data directories."""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Find existing numbered directories
    existing_dirs = glob.glob(str(logs_dir / "[0-9][0-9][0-9]"))
    if not existing_dirs:
        return 1
    
    # Extract numbers and find the maximum
    numbers = []
    for dir_path in existing_dirs:
        dir_name = Path(dir_path).name
        try:
            numbers.append(int(dir_name))
        except ValueError:
            continue
    
    return max(numbers) + 1 if numbers else 1

def setup_run_directories(run_number):
    """Set up directories for this run."""
    run_id = f"{run_number:03d}"
    
    logs_dir = Path("logs") / run_id
    data_dir = Path("data") / run_id
    
    logs_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return logs_dir, data_dir

def check_requirements():
    """Check if required dependencies are available."""
    print("Checking requirements...")
    
    # Check if model exists (relative to parent directory)
    model_path = Path("../models/nightmedia/gpt-oss-20b-q4-hi-mlx")
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



def run_neuroscope_demo(data_dir):
    """Run the NeuroScope REST interface demo."""
    print("\n" + "="*60)
    print("RUNNING NEUROSCOPE REST INTERFACE DEMO")
    print("="*60)
    
    try:
        # Set environment variable for data directory
        env = os.environ.copy()
        env['NEUROSCOPE_DATA_DIR'] = str(data_dir)
        
        result = subprocess.run([
            sys.executable, "demo_neuroscope_rest_interface.py"
        ], capture_output=False, text=True, timeout=60, env=env)
        
        # Output is displayed in real-time, no need to print captured output
        
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
            sys.executable, "../code/neuroscope_api_reference.py"
        ], capture_output=False, text=True, timeout=30)
        
        # Output is displayed in real-time, no need to print captured output
        
        return result.returncode == 0
        
    except Exception as e:
        print(f"❌ Error showing API reference: {e}")
        return False

def main():
    """Main demo runner."""
    # Set up run directories
    run_number = get_next_run_number()
    logs_dir, data_dir = setup_run_directories(run_number)
    
    # Set up logging to file
    log_file = logs_dir / "run_neuroscope_demo_py.log"
    
    # Create a custom print function that writes to both console and file
    original_stdout = sys.stdout
    
    class TeeOutput:
        def __init__(self, file_path):
            self.terminal = original_stdout
            self.log_file = open(file_path, 'w')
        
        def write(self, message):
            self.terminal.write(message)
            self.log_file.write(message)
            self.log_file.flush()
        
        def flush(self):
            self.terminal.flush()
            self.log_file.flush()
        
        def close(self):
            self.log_file.close()
    
    tee = TeeOutput(log_file)
    sys.stdout = tee
    
    try:
        print("NeuroScope MLX Engine Demo Runner")
        print("=" * 50)
        print("This demo elaborates on test_gpt_oss_20b.py by showing")
        print("how NeuroScope will interact with the MLX Engine REST API.")
        print("=" * 50)
        print(f"Script: {__file__}")
        print(f"Run: {run_number:03d}")
        print(f"Logs: {logs_dir}")
        print(f"Data: {data_dir}")
        print("=" * 50)
        
        # Check requirements
        if not check_requirements():
            print("\n❌ Requirements check failed!")
            print("Please install dependencies and ensure model is available.")
            return 1
        
        # Run tests in sequence
        tests = [
            ("NeuroScope Demo", lambda: run_neuroscope_demo(data_dir)),
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
        print(f"\n⚠️  {total_tests - successful_tests} out of {total_tests} demos failed.")
        print("Check the output above for details.")
        
        # List specific failures
        failed_tests = [name for name, success in results.items() if not success]
        if failed_tests:
            print("\nFailed tests:")
            for test_name in failed_tests:
                print(f"- ❌ {test_name}")
        
        print("\nCommon issues:")
        print("- Model not found (download required)")
        print("- Missing dependencies (pip install flask mlx)")
        print("- Memory limitations (reduce model size)")
        print("- Activation hooks endpoint not implemented")
        print("- Missing analysis configuration variables")
        
        return 1
    
    finally:
        # Restore original stdout and close log file
        sys.stdout = original_stdout
        tee.close()
        print(f"\n📝 Log saved to: {log_file}")
        print(f"📊 Data saved to: {data_dir}")

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