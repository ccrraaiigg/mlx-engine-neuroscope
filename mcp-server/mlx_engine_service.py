#!/usr/bin/env python3
"""
MLX Engine Background Service for MCP Server

Provides a persistent MLX Engine API server that runs in the background
to serve the MCP server's requests for mechanistic interpretability analysis.
"""

import sys
import os
import signal
import threading
import time
import logging
from pathlib import Path
from typing import Optional

# Add the parent directory to Python path to access mlx_engine
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from mlx_engine.api_server import MLXEngineAPI

# Configure logging to both file and console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('mlx_engine_service.log', mode='w'),  # Overwrite each time
        logging.StreamHandler()
    ],
    force=True  # Override any existing logging configuration
)
logger = logging.getLogger(__name__)

# Test logging immediately
logger.info("="*50)
logger.info("MLX Engine Service module loaded")
logger.info(f"PID: {os.getpid()}")
logger.info(f"Working directory: {os.getcwd()}")
logger.info("="*50)

class MLXEngineService:
    """Background service for MLX Engine API server."""
    
    def __init__(self, host: str = "127.0.0.1", port: int = 50111, model_path: Optional[str] = None):
        self.host = host
        self.port = port
        self.model_path = model_path
        self.api_server = None
        self.server_thread = None
        self.is_running = False
        self.pid_file = Path("mlx_engine_service.pid")
        
    def _check_existing_instance(self):
        """Check if another instance is already running."""
        if self.pid_file.exists():
            try:
                with open(self.pid_file, 'r') as f:
                    old_pid = int(f.read().strip())
                
                # Check if the process is still running
                try:
                    os.kill(old_pid, 0)  # Signal 0 just checks if process exists
                    logger.warning(f"MLX Engine service already running with PID {old_pid}")
                    return True
                except OSError:
                    # Process not running, remove stale PID file
                    logger.info(f"Removing stale PID file for PID {old_pid}")
                    self.pid_file.unlink()
                    return False
            except (ValueError, FileNotFoundError):
                # Invalid or missing PID file
                if self.pid_file.exists():
                    self.pid_file.unlink()
                return False
        return False
    
    def _write_pid_file(self):
        """Write the current PID to the PID file."""
        with open(self.pid_file, 'w') as f:
            f.write(str(os.getpid()))
        logger.info(f"PID file written: {self.pid_file}")
    
    def _remove_pid_file(self):
        """Remove the PID file."""
        if self.pid_file.exists():
            self.pid_file.unlink()
            logger.info("PID file removed")
    
    def start(self):
        """Start the MLX Engine service."""
        logger.info("Starting MLX Engine Service...")
        logger.info(f"Host: {self.host}")
        logger.info(f"Port: {self.port}")
        
        # Check if another instance is already running
        if self._check_existing_instance():
            logger.error("Another MLX Engine service instance is already running. Exiting.")
            sys.exit(1)
        
        # Write PID file
        self._write_pid_file()
        
        try:
            # Create API server instance
            self.api_server = MLXEngineAPI()
            
            # Start server in a separate thread
            self.server_thread = threading.Thread(
                target=self._run_server,
                daemon=True
            )
            self.server_thread.start()
            
            # Wait a moment for server to start
            time.sleep(2)
            
            # Optionally pre-load a model
            if self.model_path:
                self._preload_model()
            
            self.is_running = True
            logger.info("✅ MLX Engine Service started successfully")
            logger.info(f"🌐 API available at http://{self.host}:{self.port}")
            logger.info("🔧 Ready to serve MCP server requests")
            
        except Exception as e:
            logger.error(f"❌ Failed to start MLX Engine Service: {e}")
            raise
    
    def _run_server(self):
        """Run the Flask server in a separate thread."""
        try:
            self.api_server.run(host=self.host, port=self.port, debug=False)
        except Exception as e:
            logger.error(f"Server thread error: {e}")
            self.is_running = False
    
    def _preload_model(self):
        """Pre-load a model if specified."""
        try:
            logger.info(f"Pre-loading model from {self.model_path}...")
            # AGENT.md: Never fake anything. Model loading will be handled via real API calls from MCP server
            logger.info("Model loading will be handled via API calls when needed - no placeholder pre-loading")
        except Exception as e:
            logger.warning(f"Model path configuration failed: {e}")
    
    def stop(self):
        """Stop the MLX Engine service."""
        logger.info("Stopping MLX Engine Service...")
        self.is_running = False
        
        # Remove PID file
        self._remove_pid_file()
        
        if self.api_server:
            # Flask doesn't have a clean shutdown method in this context
            # The process will be terminated by the signal handler
            logger.info("Service stopped")
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal."""
        try:
            while self.is_running:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()

def signal_handler(signum, frame):
    """Handle shutdown signals."""
    logger.info(f"Received signal {signum}")
    global service
    if service:
        service.stop()
    # Ensure PID file is removed even if service.stop() fails
    pid_file = Path("mlx_engine_service.pid")
    if pid_file.exists():
        pid_file.unlink()
        logger.info("PID file cleaned up on signal")
    sys.exit(0)

def main():
    """Main service entry point."""
    global service
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Default model path (can be overridden via environment variable)
    default_model_path = "../models/nightmedia/gpt-oss-20b-q4-hi-mlx"
    model_path = os.environ.get('MLX_MODEL_PATH', default_model_path)
    
    # Check if model exists
    if os.path.exists(model_path):
        logger.info(f"Model found at {model_path}")
    else:
        logger.warning(f"Model not found at {model_path}")
        model_path = None
    
    # Create and start service
    service = MLXEngineService(
        host="127.0.0.1",
        port=50111,
        model_path=model_path
    )
    
    try:
        service.start()
        
        logger.info("MLX Engine Service is running...")
        logger.info("Press Ctrl+C to stop")
        
        # Keep the service running
        service.wait_for_shutdown()
        
    except Exception as e:
        logger.error(f"Service error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
