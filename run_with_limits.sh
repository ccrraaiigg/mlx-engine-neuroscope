#!/bin/bash

# Script to run tests with memory limits and monitoring

echo "Setting up memory limits and monitoring..."

# Set memory limit (adjust based on your system)
# This limits virtual memory to ~18GB to prevent system crash
ulimit -v 18000000

echo "Memory limit set to: $(ulimit -v) KB"
echo "Available memory:"
free -h 2>/dev/null || vm_stat | head -5

echo ""
echo "Starting test with memory monitoring..."
echo "Press Ctrl+C to stop if memory usage gets too high"
echo ""

# Run the test with timeout to prevent hanging
timeout 300 python test_gpt_oss_20b.py

exit_code=$?

if [ $exit_code -eq 124 ]; then
    echo ""
    echo "⚠ Test timed out after 5 minutes"
    echo "This suggests the model is too large for available memory"
elif [ $exit_code -ne 0 ]; then
    echo ""
    echo "✗ Test failed with exit code: $exit_code"
else
    echo ""
    echo "✓ Test completed successfully"
fi

echo ""
echo "Final memory status:"
free -h 2>/dev/null || vm_stat | head -5