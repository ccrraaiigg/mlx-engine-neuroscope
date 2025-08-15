# Cosmos Graph Visualization Integration

This directory contains the foundation for integrating Cosmos Graph, a high-performance graph visualization library, into the MLX Engine with NeuroScope Integration project.

## 🏗️ Foundation Components

### Core Modules

- **`types/graph_types.js`** - TypeScript-style interfaces and type definitions for graph data structures
- **`utils/webgl_capabilities.js`** - WebGL capability detection and fallback mechanisms
- **`renderer/cosmos_graph_renderer.js`** - High-performance WebGL-based graph renderer using Cosmos Graph
- **`converter/graph_converter.js`** - Converts MLX Engine data to Cosmos Graph-compatible format
- **`generator/visualization_generator.js`** - Orchestrates data fetching and graph generation

### Test Files

- **`test_foundation.js`** - Comprehensive test suite for all foundation components
- **`test_browser.html`** - Interactive browser test page for WebGL and visualization testing

## 🚀 Getting Started

### Prerequisites

1. **Node.js Runtime** - The project uses Node.js for JavaScript execution
2. **Modern Browser** - WebGL support required for optimal performance
3. **Cosmos Graph Library** - Will be installed via npm (configured in `package.json`)

### Installation

The Cosmos Graph library is configured in `mcp-server/package.json`:

```json
{
  "dependencies": {
    "@cosmos.gl/graph": "^2.3.1"
  }
}
```

To install dependencies:

```bash
cd mcp-server
npm install
```

### Running Tests

**Command Line Tests:**
```bash
cd mcp-server
npm run test:foundation
# or directly: node src/visualization/test_foundation.js
```

**Browser Tests:**
```bash
cd mcp-server
npm run test:browser
# or directly: node src/visualization/serve_test.js
# Then open the test page in your browser:
#   http://localhost:8081/src/visualization/test_browser.html
```

**Sample Graph Demo:**
```bash
cd mcp-server
npm run demo:sample  # Command-line demo of sample graph structure
npm run test:sample  # Detailed test of sample graph functionality
```

### Interactive Browser Testing

The browser test page (`test_browser.html`) provides comprehensive testing of all visualization capabilities:

- ✅ **WebGL capabilities detection** - Tests browser support and performance level
- ✅ **Graph data conversion** - Tests circuit, attention, and flow data conversion
- ✅ **WebGL Graph Rendering** - Hardware-accelerated graph visualization with @cosmos.gl/graph
- ✅ **Interactive features** - Click nodes, hover for details, zoom/pan controls
- ✅ **Physics simulation** - Real-time node positioning and layout randomization
- ✅ **Export functionality** - Downloads high-quality PNG images
- ✅ **Canvas 2D fallback** - Automatic fallback for unsupported browsers
- ✅ **SVG fallback** - Static visualization for maximum compatibility

#### Sample Neural Circuit

The "Load Sample Graph" feature creates a 10-node neural network circuit:

- **Input Layer**: Token and position inputs
- **Attention Layer**: Self-attention and cross-attention heads  
- **Processing Layer**: Feature detectors and pattern matchers
- **Circuit Layer**: IOI and induction circuits
- **Output Layer**: Logit output

**Automatic rendering mode selection:**

1. **Cosmos Graph WebGL** (primary):
   - Hardware-accelerated WebGL rendering via local npm package
   - Real-time physics simulation with interactive controls
   - Click nodes for details, hover for labels
   - Smooth zoom, pan, and layout randomization
   - High-performance rendering for complex graphs

2. **Canvas 2D Fallback** (automatic):
   - Software rendering when WebGL fails
   - Static layout with color coding and legend
   - Export functionality maintained
   - Guaranteed compatibility across all browsers

## 📊 Data Structures

### Graph Data Format

```javascript
const graphData = {
  id: "unique_graph_id",
  nodes: [
    {
      id: "node_1",
      label: "Node Label",
      type: "neuron|attention_head|layer|circuit|token|feature",
      value: 0.8,  // Size/importance (0-1)
      color: "#4285f4",
      metadata: {
        layer: 0,
        activation_strength: 0.8,
        semantic_role: "input"
      }
    }
  ],
  links: [
    {
      id: "link_1",
      source: "node_1",
      target: "node_2", 
      weight: 0.6,  // Connection strength (0-1)
      type: "activation|attention|circuit|causal|similarity",
      color: "#cccccc",
      metadata: {
        connection_type: "circuit",
        causal_strength: 0.6
      }
    }
  ],
  metadata: {
    title: "Graph Title",
    description: "Graph description",
    type: "circuit|attention|activation_flow|model_architecture|comparison",
    created_at: "2024-01-01T00:00:00Z",
    model_info: {
      model_id: "gpt-oss-20b",
      architecture: "transformer",
      num_layers: 20
    }
  },
  layout: {
    algorithm: "force|hierarchical|circular",
    parameters: { /* algorithm-specific params */ }
  },
  styling: {
    theme: "light|dark|custom",
    colorScheme: ["#4285f4", "#34a853", "#fbbc04", "#ea4335"],
    nodeScale: [3, 15],
    linkScale: [1, 5]
  }
}
```

## 🎯 Core Features

### WebGL Capability Detection

```javascript
import { WebGLCapabilities } from './utils/webgl_capabilities.js';

// Detect browser capabilities
const capabilities = WebGLCapabilities.detectCapabilities();
console.log(capabilities);
// {
//   webgl: true,
//   webgl2: false,
//   maxTextureSize: 4096,
//   maxVertexUniforms: 256,
//   performanceLevel: 'medium'
// }

// Get optimal configuration
const config = WebGLCapabilities.getOptimalConfig(capabilities);

// Simplify large graphs for performance
const simplifiedGraph = WebGLCapabilities.simplifyGraph(largeGraph, 500);
```

### Graph Data Conversion

```javascript
import { GraphConverter } from './converter/graph_converter.js';

const converter = new GraphConverter();

// Convert circuit data
const circuitGraph = converter.convertCircuitToGraph(circuitData, activations);

// Convert attention patterns
const attentionGraph = converter.convertAttentionToGraph(attentionData);

// Convert activation flows
const flowGraph = converter.convertActivationsToGraph(activationData);
```

### Visualization Generation

```javascript
import { VisualizationGenerator } from './generator/visualization_generator.js';

const generator = new VisualizationGenerator(mlxClient);

// Generate different types of visualizations
const circuitViz = await generator.generateCircuitGraph('IOI_circuit');
const attentionViz = await generator.generateAttentionGraph([8, 12]);
const flowViz = await generator.generateActivationFlowGraph(['hello', 'world']);
```

### Cosmos Graph Rendering

```javascript
import { Graph } from '@cosmos.gl/graph';

// Initialize renderer
const container = document.getElementById('graph-container');
const graph = new Graph(container, {
  pointSize: 6,
  linkWidth: 2,
  backgroundColor: '#ffffff',
  showLabels: true
});

// Load and render graph
await renderer.initialize();
await renderer.loadGraph(graphData);

// Set up interactions
renderer.onNodeClick((node) => {
  console.log('Clicked node:', node);
});

renderer.onNodeHover((node) => {
  if (node) {
    console.log('Hovering over:', node.label);
  }
});
```

## 🔧 Configuration

### Performance Optimization

The system automatically optimizes based on detected browser capabilities:

- **High Performance**: Full WebGL 2.0 with all features enabled
- **Medium Performance**: WebGL 1.0 with reduced complexity
- **Low Performance**: Simplified graphs with fallback rendering

### Dark Theme Support

The visualization system uses a modern dark theme optimized for developer workflows:

- **GitHub-inspired color palette**: Uses GitHub's dark theme colors for consistency
- **High contrast**: Ensures readability and accessibility
- **WebGL-optimized**: Dark backgrounds reduce eye strain during long analysis sessions
- **SVG fallback**: Dark theme extends to fallback visualizations

### Fallback Mechanisms

1. **WebGL → Canvas 2D**: If WebGL fails, falls back to Canvas 2D rendering
2. **Canvas → SVG**: If Canvas fails, generates static SVG visualization
3. **Graph Simplification**: Automatically reduces complexity for large graphs

## 🧪 Testing

### Foundation Test Results

```bash
✅ Test 1: Type definitions
✅ Test 2: WebGL capabilities detection  
✅ Test 3: Graph converter
✅ Test 4: Visualization generator
✅ Test 5: Graph data validation
✅ Test 6: Performance optimization
```

### Browser Compatibility

- **Chrome/Edge**: Full WebGL 2.0 support
- **Firefox**: WebGL 1.0 support
- **Safari**: WebGL 1.0 support with limitations
- **Mobile**: Reduced performance mode with simplification

## 🔮 Next Steps

This foundation enables the following upcoming tasks:

1. **Implement MCP Tools** - Create visualization MCP tools for chat interface
2. **MLX Engine Integration** - Connect to real MLX Engine REST API
3. **Web Interface Components** - Build React/HTML components
5. **Export Functionality** - Implement PNG/SVG/JSON export
6. **NeuroScope Bridge** - Integrate with Smalltalk interface

## 📚 References

- [Cosmos Graph Documentation](https://cosmos.uber.com/graph/)
- [WebGL Specification](https://www.khronos.org/webgl/)
- [MLX Engine API Documentation](../../../README.md)
- [MCP Server Documentation](../../README.md)

## 🐛 Troubleshooting

### Common Issues

1. **WebGL Not Supported**
   - Check browser compatibility
   - Enable hardware acceleration
   - Use SVG fallback mode

2. **Performance Issues**
   - Reduce graph complexity with filtering
   - Enable level-of-detail rendering
   - Use simplified layouts

3. **Import Errors**
   - Ensure Node.js is properly configured
   - Check import paths in `package.json`
   - Verify all dependencies are installed with `npm install`

### Debug Mode

Enable debug logging:

```javascript
// Set debug flag before importing modules
globalThis.DEBUG_VISUALIZATION = true;
```

This will provide detailed console output for troubleshooting.