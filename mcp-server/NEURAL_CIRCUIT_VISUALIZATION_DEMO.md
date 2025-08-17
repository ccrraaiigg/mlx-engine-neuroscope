# Neural Circuit Visualization Demo Guide

This guide documents how to create real-time, interactive visualizations of neural circuits from GPT-OSS-20B using the mechanistic interpretability MCP server.

## Overview

We have successfully built a complete end-to-end pipeline that:
1. **Captures real activation data** from GPT-OSS-20B during inference
2. **Converts raw activations** to interactive graph format (nodes and links)
3. **Renders beautiful WebGL2 visualizations** using Cosmos Graph
4. **Provides interactive exploration** with node hovering and real-time updates

## System Architecture

```
GPT-OSS-20B Model → MLX Engine → MCP Server → Cosmos Graph WebGL2 → Browser
     (Inference)    (Capture)   (Convert)     (Visualize)      (Interact)
```

## Key Components

### 1. MLX Engine (Port 50111)
- Runs GPT-OSS-20B model with activation hooks
- Captures neural activations during text generation
- Provides REST API endpoints for model interaction

### 2. MCP Server (`mcp-server/src/filesystem_pattern_server.js`)
- Mechanistic Interpretability MCP Server (Version 24+)
- 16 tools for neural analysis and visualization
- Real-time data processing and conversion

### 3. Cosmos Graph Visualization (`mcp-server/src/visualization/`)
- Professional WebGL2-based graph renderer
- Hardware-accelerated interactive visualizations
- Node hovering, clicking, and real-time updates

## Demo Instructions

**Important**: This demo is driven by **conversation with the user**. The agent should respond to user requests by invoking the appropriate MCP tools. The following steps represent a typical demo flow, but the agent should adapt based on user questions and interests.

### Prerequisites
- GPT-OSS-20B model loaded and running on MLX Engine (port 50111)
- MCP server connected to Cursor
- Visualization server running on port 8888

### Typical Demo Flow

When a user asks to see the neural circuit visualization demo, the agent should:

1. **Verify system status** by checking MCP server version and service health
2. **Start any missing services** if needed 
3. **Capture fresh activation data** from GPT-OSS-20B for a math problem
4. **Create interactive visualization** from the captured data
5. **Open browser** to display the WebGL2 visualization
6. **Explain the results** and answer user questions about what they're seeing

### Step 1: Verify System Status (Agent Action)

The agent should first check system status when requested:

#### Check MCP Server Version
```
mcp_mechanistic-interpretability_version
```
Expected: Version 26+ with clean responses (no JSON errors)

**Important**: The version number indicates the server state and must be incremented every time the MCP server code is modified.

#### Check All Services Health  
```
mcp_mechanistic-interpretability_health_check
service: "all"
```
Expected response:
```json
{
  "success": true,
  "overall_status": "healthy",
  "services": {
    "mlx_engine": {
      "status": "healthy",
      "component": "MLX Engine REST API",
      "version": "1.2.0",
      "current_model": "gpt-oss-20b",
      "ready": true
    },
    "visualization_server": {
      "status": "healthy", 
      "component": "Visualization Web Server",
      "version": "1.1.0",
      "port": 8888,
      "ready": true
    }
  }
}
```

#### Start Services (if needed)
If any service is unreachable, the agent should start it:

**Start MLX Engine:**
```
mcp_mechanistic-interpretability_start_server
service: "mlx"
```

**Start Visualization Server:**
```
mcp_mechanistic-interpretability_start_server  
service: "visualization"
```

### Step 2: Capture Fresh Activation Data (Agent Action)

The agent should ask the user for a math problem or choose one, then capture activations:

```
mcp_mechanistic-interpretability_capture_activations
prompt: "What is [X] × [Y]?"  # Use different math problems each time
max_tokens: 50
temperature: 0.1
```

**User Interaction**: The agent can ask the user: *"What math problem would you like to see the neural circuits for? For example, multiplication, addition, or division?"*

This captures real neural activations from:
- `model.layers.0.mlp.mlp` (red nodes - Multi-Layer Perceptron)
- `model.layers.0.self_attn.attention` (blue nodes - Attention mechanism)
- `model.layers.5.self_attn.attention` (blue nodes - Higher-layer attention)

Each activation contains:
- `component`: "mlp" or "attention" 
- `layer_name`: e.g., "model.layers.0.mlp"
- `shape`: Tensor dimensions [1, 32, 768]
- `dtype`: "float32"

### Step 3: Create Interactive Visualization (Agent Action)

The agent automatically uses the captured data to create the visualization:

```
mcp_mechanistic-interpretability_circuit_diagram
circuit_data: {raw activation data from step 2}
circuit_name: "GPT-OSS-20B [Problem Type] Circuit ([specific problem])"
```

This automatically:
- Converts raw activations to nodes and links
- Generates WebGL2-ready graph data
- Creates interactive HTML visualization
- Saves files to `mcp-server/src/visualization/`

### Step 4: Open Browser Visualization (Agent Action)

The agent opens the visualization in the user's browser:

```
mcp_mechanistic-interpretability_open_browser
url: "http://localhost:8888/real_circuit.html"
```

**User Interaction**: The agent should explain: *"I've opened the interactive visualization in your browser. You can hover over the nodes to see layer details and explore how the neural network processes the math problem."*

## Expected Results

### What the User Will See

The agent should explain what's happening as the demo progresses and help the user understand the visualization.

### Console Output (Browser DevTools)
```
🔥 Initializing Cosmos Graph WebGL2 visualization
Raw activation data keys: Array(4) 
Data conversion status: Already converted
Expected nodes: 3
Expected links: 2
✅ Using converted graph data
✅ Cosmos Graph initialized
📊 Loading graph data...
✅ Graph loaded successfully
🚀 Cosmos Graph WebGL2 visualization ready
```

### Interactive Features
- **Node Hovering**: Shows layer names (e.g., "model.layers.0.mlp (mlp)")
- **Color Coding**: Red for MLP components, Blue for attention
- **Real Metadata**: Actual tensor shapes and component counts
- **Live Updates**: Node/link counts update in real-time

### Visual Elements
- **3 Nodes**: Representing the captured neural components
- **2 Links**: Showing activation flow between layers
- **WebGL2 Rendering**: Hardware-accelerated smooth graphics
- **Responsive Layout**: Adapts to browser window size

## Technical Details

### Data Flow
1. **Raw Activations**: Captured from specific model layers during inference
2. **Server-side Conversion**: Transforms activations into `{nodes: [], links: []}` format
3. **Browser-side Rendering**: Cosmos Graph processes data for WebGL2 display
4. **Interactive Updates**: Real-time hover and click event handling

### Node Structure
```javascript
{
  id: "node_0",
  label: "model.layers.0.mlp (mlp)",
  type: "mlp", // or "attention"
  value: 0.8,
  color: [1.0, 0.4, 0.4, 1.0], // Red for MLP, Blue for attention
  layer: 0,
  position: { x: 25, y: 15 },
  metadata: {
    shape: [1, 32, 768],
    count: 46,
    component: "mlp",
    layer_name: "model.layers.0.mlp"
  }
}
```

### Link Structure
```javascript
{
  id: "link_0",
  source: "node_0",
  target: "node_1", 
  weight: 0.8,
  color: [1, 1, 1, 0.6],
  type: "activation_flow",
  metadata: {
    connection_type: "activation_flow"
  }
}
```

## Logging Architecture

### Server Logs (`mcp-server/log`)
```
2025-08-16 11:38:23.821 [info] Mechanistic Interpretability MCP Server v24 running
2025-08-16 11:38:23.822 [info] Registered 16 tools for neural analysis
```

### Browser Console (DevTools)
```javascript
console.log('🔥 Initializing Cosmos Graph WebGL2 visualization');
console.log('Expected nodes:', 3);
console.log('✅ Graph loaded successfully');
```

### MCP Responses (JSON only)
```json
{
  "success": true,
  "nodes_count": 3,
  "edges_count": 2,
  "visualization_url": "http://localhost:8888/real_circuit.html"
}
```

## Troubleshooting

### Common Issues

1. **JSON Parsing Errors**: Fixed in v22+ by removing console.log from tool responses
2. **Variable Scoping**: Fixed in v24 by using `rawCircuitData` consistently  
3. **Empty Visualizations**: Ensure data conversion detects `model.layers.*` keys
4. **Browser Errors**: Check that Cosmos Graph renderer is properly imported

### Verification Steps

1. **Check MCP Server**: `mcp_mechanistic-interpretability_version` → Should be 26+
2. **Check All Services**: `mcp_mechanistic-interpretability_health_check` → All should be healthy  
3. **Manual MLX Engine**: `curl http://localhost:50111/health` → Should return JSON with version
4. **Manual Visualization**: `curl http://localhost:8888/health` → Should return JSON with version
5. **Test Data Capture**: Activation arrays should contain multiple tensors
6. **Browser Console**: Should show successful Cosmos Graph initialization

### Health Check Tools

#### Check Individual Services
```bash
# Check only MLX Engine
mcp_mechanistic-interpretability_health_check service: "mlx"

# Check only Visualization Server  
mcp_mechanistic-interpretability_health_check service: "visualization"

# Check all services (default)
mcp_mechanistic-interpretability_health_check service: "all"
```

#### Start Servers Automatically
```bash
# Start MLX Engine if not running
mcp_mechanistic-interpretability_start_server service: "mlx"

# Start Visualization Server if not running
mcp_mechanistic-interpretability_start_server service: "visualization" 

# Force restart (even if already running)
mcp_mechanistic-interpretability_start_server service: "mlx" force: true
```

#### Health Response Format
```json
{
  "success": true,
  "overall_status": "healthy|degraded",
  "timestamp": "2025-08-16T18:45:39.133Z",
  "services": {
    "service_name": {
      "status": "healthy|unhealthy|unreachable",
      "reachable": true|false,
      "component": "Service Description",
      "version": "1.2.0",
      "timestamp": "2025-08-16T18:45:39.133Z",
      "ready": true|false,
      "error": "Error message if applicable"
    }
  }
}
```

## Critical Development Workflow

### Code Modification Protocol

**MANDATORY**: Every time you modify MCP server code (`mcp-server/src/filesystem_pattern_server.js`):

1. **Increment Version Number**: Update the version in the `versionTool` function
   ```javascript
   version: 25, // Increment from previous number
   changes: "Description of what was changed"
   ```

2. **Ask User to Reload**: Always request user to reload the MCP server in Cursor
   ```
   "Please reload the mcp-server in Cursor"
   ```

3. **Verify Reload**: First tool call after reload MUST be `version` to confirm changes
   ```
   mcp_mechanistic-interpretability_version
   ```

4. **Confirm Success**: Check that the returned version matches your increment
   - ✅ Version matches → Changes applied successfully
   - ❌ Version doesn't match → Server not reloaded, ask user to reload again

### Example Development Cycle
```
1. Modify code → version: 24 → version: 25
2. "Please reload the mcp-server in Cursor"
3. Call version tool → Expect: {"version": 25}
4. If version ≠ 25 → "Please reload again"
5. If version = 25 → Continue with testing
```

**Why This Matters**: The version number is the only reliable way to confirm that code changes have been loaded by the MCP server. Without this verification, you may be testing old code and debugging non-existent issues.

## File Locations

- **MCP Server**: `mcp-server/src/filesystem_pattern_server.js`
- **MLX Engine Service**: `mcp-server/mlx_engine_service.py`  
- **MLX Engine API**: `mlx_engine/api_server.py`
- **Visualization Server**: `mcp-server/src/visualization/server.js`
- **Visualization Data**: `mcp-server/src/visualization/real_circuit_data.json`
- **Interactive HTML**: `mcp-server/src/visualization/real_circuit.html`
- **Cosmos Renderer**: `mcp-server/src/visualization/renderer/cosmos_graph_renderer.js`
- **Server Logs**: `mcp-server/log`

## Health Endpoints

- **MLX Engine**: `http://localhost:50111/health`
- **Visualization Server**: `http://localhost:8888/health`

Example health response:
```json
{
  "status": "healthy",
  "service": "mlx-engine-neuroscope", 
  "component": "MLX Engine REST API",
  "version": "1.2.0",
  "timestamp": "2025-08-16T18:45:39.133Z",
  "current_model": "gpt-oss-20b",
  "ready": true
}
```

## Demo Variations

The agent should adapt to user interests and questions:

### Different Problem Types
Based on user preference:
- **Arithmetic**: "What is 47 × 83?"
- **Addition**: "Calculate 127 + 256"  
- **Multiplication**: "What is 89 × 17?"
- **User's Choice**: "What math problem interests you?"

### Interactive Exploration  
The agent can respond to user questions like:
- *"What do the red vs blue nodes represent?"*
- *"How does the network solve multiplication differently than addition?"*
- *"Can we see deeper layers of the network?"*
- *"What happens with a more complex problem?"*

### Layer Exploration
- Modify capture hooks to explore different layers (0-19)
- Compare attention vs MLP activation patterns
- Study information flow across transformer layers

### Advanced Features
For users wanting deeper analysis:
- Real-time activation streaming
- Multi-layer circuit discovery
- Attention pattern visualization
- MLP decomposition analysis

## Success Criteria

✅ **Clean MCP Communication**: No JSON parsing errors  
✅ **Real Data Flow**: Actual GPT-OSS-20B activations captured  
✅ **Interactive Visualization**: WebGL2 rendering with hover effects  
✅ **Proper Logging**: Server events to log file, debugging to browser console  
✅ **Professional UI**: Beautiful, responsive graph visualization  
✅ **Version Control**: Proper version incrementing and reload verification  
✅ **Health Monitoring**: Comprehensive service health checks and auto-startup  
✅ **Server Management**: Automated MLX Engine and Visualization server control  

## Next Steps

1. **Expand Layer Coverage**: Capture from multiple transformer layers
2. **Attention Patterns**: Visualize attention weight matrices
3. **Real-time Streaming**: Live activation updates during generation
4. **Circuit Discovery**: Automated identification of computational circuits
5. **Comparative Analysis**: Side-by-side circuit comparisons

---

*This system represents a complete pipeline for mechanistic interpretability research, enabling real-time exploration of neural network internals with professional-grade visualization tools.*
