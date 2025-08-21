#!/bin/bash

# MLX Engine Service Startup Script
# Starts the MLX Engine API server in the background for MCP server

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SERVICE_SCRIPT="$SCRIPT_DIR/mlx_engine_service.py"
PID_FILE="$SCRIPT_DIR/mlx_engine_service.pid"
LOG_FILE="$SCRIPT_DIR/mlx_engine_service.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[MLX Service]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[MLX Service]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[MLX Service]${NC} $1"
}

print_error() {
    echo -e "${RED}[MLX Service]${NC} $1"
}

check_dependencies() {
    print_status "Checking dependencies..."
    
    # Check Python3
    if ! command -v python3 &> /dev/null; then
        print_error "python3 is required but not found"
        exit 1
    fi
    
    # Check if we can import mlx_engine
    cd "$SCRIPT_DIR/.."
    if ! python3 -c "import mlx_engine" 2>/dev/null; then
        print_error "MLX Engine module not found. Make sure you're in the correct directory."
        exit 1
    fi
    
    print_success "Dependencies check passed"
}

is_running() {
    if [ -f "$PID_FILE" ]; then
        local pid=$(cat "$PID_FILE")
        if ps -p "$pid" > /dev/null 2>&1; then
            return 0
        else
            # PID file exists but process is not running
            rm -f "$PID_FILE"
            return 1
        fi
    fi
    return 1
}

start_service() {
    if is_running; then
        print_warning "MLX Engine service is already running (PID: $(cat "$PID_FILE"))"
        return 0
    fi
    
    print_status "Starting MLX Engine service..."
    check_dependencies
    
    # Clean up old log file
    if [ -f "$LOG_FILE" ]; then
        rm -f "$LOG_FILE"
        print_status "Cleaned up old log file"
    fi
    
    # Set PYTHONPATH and start the service
    cd "$SCRIPT_DIR"
    export PYTHONPATH="$SCRIPT_DIR/..:$PYTHONPATH"
    
    # Start in background and save PID
    nohup python3 "$SERVICE_SCRIPT" > "$LOG_FILE" 2>&1 &
    local pid=$!
    echo $pid > "$PID_FILE"
    
    # Wait a moment and check if it started successfully
    sleep 3
    if is_running; then
        print_success "MLX Engine service started successfully"
        print_success "PID: $pid"
        print_success "Log: $LOG_FILE"
        print_success "API available at: http://127.0.0.1:50111"
        
        # Test the health endpoint
        if command -v curl &> /dev/null; then
            print_status "Testing health endpoint..."
            if curl -s "http://127.0.0.1:50111/health" > /dev/null; then
                print_success "✅ Health check passed"
            else
                print_warning "⚠️  Health check failed (service may still be starting)"
            fi
        fi
    else
        print_error "Failed to start MLX Engine service"
        if [ -f "$LOG_FILE" ]; then
            print_error "Check log file: $LOG_FILE"
            tail -10 "$LOG_FILE"
        fi
        exit 1
    fi
}

stop_service() {
    if ! is_running; then
        print_warning "MLX Engine service is not running"
        return 0
    fi
    
    local pid=$(cat "$PID_FILE")
    print_status "Stopping MLX Engine service (PID: $pid)..."
    
    # Send SIGTERM
    kill "$pid" 2>/dev/null || true
    
    # Wait for graceful shutdown
    local count=0
    while is_running && [ $count -lt 10 ]; do
        sleep 1
        count=$((count + 1))
    done
    
    # Force kill if still running
    if is_running; then
        print_warning "Forcing shutdown..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
    
    if ! is_running; then
        rm -f "$PID_FILE"
        print_success "MLX Engine service stopped"
    else
        print_error "Failed to stop MLX Engine service"
        exit 1
    fi
}

status_service() {
    if is_running; then
        local pid=$(cat "$PID_FILE")
        print_success "MLX Engine service is running (PID: $pid)"
        print_success "API available at: http://127.0.0.1:50111"
        
        # Test health endpoint
        if command -v curl &> /dev/null; then
            if curl -s "http://127.0.0.1:50111/health" > /dev/null; then
                print_success "✅ Health check: PASS"
            else
                print_warning "⚠️  Health check: FAIL"
            fi
        fi
    else
        print_warning "MLX Engine service is not running"
    fi
}

show_logs() {
    if [ -f "$LOG_FILE" ]; then
        print_status "Showing last 20 lines of log file:"
        echo "----------------------------------------"
        tail -20 "$LOG_FILE"
        echo "----------------------------------------"
        print_status "Full log: $LOG_FILE"
    else
        print_warning "No log file found: $LOG_FILE"
    fi
}

case "${1:-start}" in
    start)
        start_service
        ;;
    stop)
        stop_service
        ;;
    restart)
        stop_service
        start_service
        ;;
    status)
        status_service
        ;;
    logs)
        show_logs
        ;;
    *)
        echo "Usage: $0 {start|stop|restart|status|logs}"
        echo ""
        echo "Commands:"
        echo "  start   - Start the MLX Engine service"
        echo "  stop    - Stop the MLX Engine service"
        echo "  restart - Restart the MLX Engine service"
        echo "  status  - Show service status"
        echo "  logs    - Show recent log entries"
        exit 1
        ;;
esac
