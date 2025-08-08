"""
Script to start the MLX Engine API server on a specific port.
"""
from mlx_engine.api_server import MLXEngineAPI

def main():
    # Create and run the API server
    api = MLXEngineAPI()
    print("Starting MLX Engine API server on 127.0.0.1:50111")
    api.run(host='127.0.0.1', port=50111, debug=True)

if __name__ == '__main__':
    main()
