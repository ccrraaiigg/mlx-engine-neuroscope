# Mechanistic Interpretability MCP Server

An interactive chatbot and MCP server for mechanistic interpretability analysis with MLX Engine integration and beautiful visualizations.

## 🚨 CRITICAL: MCP Server Restart Requirement

**MANDATORY FOR ALL DEVELOPMENT**: When modifying MCP server code, the server MUST be restarted for changes to take effect.

### Quick Restart Protocol:
1. Make code changes + increment version number
2. Ask user to restart MCP server in IDE
3. Verify restart with `version` tool call
4. Restart demo workflow from beginning

📋 **See [MCP_RESTART_CHECKLIST.md](./MCP_RESTART_CHECKLIST.md) for detailed protocol**

## 🚀 Quick Start

### Prerequisites

1. **Node.js 18+** - Required for running the server
2. **Anthropic API Key** - For the chatbot functionality (already in `keys` file)
3. **MLX Engine** - Required for real model analysis (must be running)

### Installation

```bash
# Clone and navigate to the mcp-server directory
cd mcp-server

# Install dependencies
npm install

# Start the MLX Engine background service
./start_mlx_service.sh start

# Check that the service is running
./start_mlx_service.sh status

# API keys are already configured in the 'keys' file
```

### Running the Interactive Chatbot

```bash
# Start the interactive chatbot
npm run chat

# Or run a demo session
npm run demo

# Test the chatbot functionality
npm run chat:test

# Alternative command
npm run chatbot
```

The chatbot will automatically:
- ✅ **Check MCP server version** (ensures version 25+)
- ✅ **Verify service health** (MLX Engine + Visualization server)
- ✅ **Load demo documentation** into its knowledge base
- ✅ **Connect to real backend services** (same as Cursor uses)

### Example Conversation

```
🧠 MI-Chat> discover circuits for indirect object identification in GPT-OSS-20B

🤖 I'll help you discover circuits for indirect object identification (IOI) in GPT-OSS-20B! 
This is a fascinating phenomenon where the model learns to identify which entity should 
receive an indirect object in sentences like "John and Mary went to the store, John gave a drink to..."

🔧 Executing core_discover_circuits...
✅ Tool execution completed:
{
  "success": true,
  "circuits": [
    {
      "id": "circuit_001",
      "name": "IOI_primary_circuit",
      "confidence": 0.85,
      "layers": [8, 9, 10],
      "components": ["attention_head_8_3", "mlp_9", "attention_head_10_1"]
    }
  ]
}

💡 Tip: I can create visualizations of this data! Try asking me to "visualize the results"

🧠 MI-Chat> visualize the circuit

🤖 I'll create a beautiful interactive visualization of the IOI circuit for you!

🔧 Executing viz_circuit_diagram...
✅ Tool execution completed:
{
  "success": true,
  "visualization_url": "http://localhost:8888/index.html?data=circuit_graph.json",
  "nodes_count": 5,
  "links_count": 4
}

🎨 Opening visualization: http://localhost:8888/index.html?data=circuit_graph.json
```

## 🎯 Available Commands

### System Commands
- `help` - Show available commands and examples
- `status` - Display system status and configuration
- `tools` - List all available analysis tools
- `history` - Show conversation history
- `clear` - Clear screen
- `viz` - Open visualization interface
- `exit` / `quit` - Exit the chatbot

### Analysis Examples
- "Discover circuits for indirect object identification in GPT-OSS-20B"
- "Analyze mathematical reasoning in the model"
- "Show me attention patterns in layer 8"
- "Find neurons that activate for factual recall"
- "Visualize the circuit diagram"

## 🛠️ Available Tools

### Core Analysis Tools
- `core_discover_circuits` - Discover circuits for specific phenomena
- `core_localize_features` - Localize neurons for specific features
- `ping` - Test connectivity

### MLX Engine Integration Tools
- `mlx_load_model` - Load models in MLX Engine
- `mlx_create_hooks` - Create activation hooks
- `mlx_capture_activations` - Capture activations during generation
- `mlx_analyze_math` - Analyze mathematical reasoning circuits
- `mlx_analyze_attention` - Analyze attention patterns
- `mlx_analyze_factual` - Analyze factual recall circuits
- `mlx_track_residual` - Track residual stream flow
- `mlx_export_neuroscope` - Export to NeuroScope format

### Visualization Tools
- `viz_circuit_diagram` - Create interactive circuit diagrams
- `viz_attention_patterns` - Visualize attention patterns
- `viz_activation_flow` - Show activation flow through layers
- `viz_open_browser` - Open visualization interface
- `viz_generate_report` - Generate comprehensive analysis reports

## 🎨 Visualization System

The system includes a powerful WebGL-based visualization system using Cosmos Graph:

### Features
- **Interactive Circuit Diagrams** - Click nodes, drag to explore
- **Attention Pattern Visualization** - See how attention flows
- **Activation Flow Graphs** - Track information through layers
- **Real-time Physics** - Dynamic force-directed layouts
- **Export Capabilities** - Save visualizations as PNG/SVG
- **Fallback Support** - Works even without WebGL

### Accessing Visualizations
1. **Automatic** - Visualizations open automatically after analysis
2. **Manual** - Type `viz` in the chatbot
3. **Direct** - Visit http://localhost:8888 in your browser

## 🔧 Configuration

### Environment Variables
```bash
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key-here

# Optional
MLX_ENGINE_API_URL=http://localhost:50111
MLX_ENGINE_API_KEY=optional-api-key
MCP_PORT=3000
LOG_LEVEL=INFO
```

### Configuration File
Create `config.json` for advanced configuration:
```json
{
  "mcp": {
    "port": 3000,
    "host": "localhost"
  },
  "mlxEngine": {
    "apiUrl": "http://localhost:50111",
    "timeout": 30000
  },
  "anthropic": {
    "model": "claude-3-sonnet-20240229"
  },
  "logging": {
    "level": "INFO",
    "format": "text"
  }
}
```

## 🧪 Development

### Running Tests
```bash
npm test
```

### Starting Individual Components
```bash
# Start just the visualization server
npm run viz

# Start the MCP server (without chatbot)
npm start

# Development mode with auto-reload
npm run dev
```

### MLX Engine Connection
The system connects directly to the MLX Engine REST API:
- **Real Connection** - All analysis uses actual MLX Engine data
- **Failure Handling** - Clear error messages when MLX Engine is unavailable
- **No Mock Data** - Failures will show exactly what needs to be implemented

## 📊 Example Analysis Workflow

1. **Start the chatbot**: `npm run chat`
2. **Load a model**: "Load GPT-OSS-20B model"
3. **Discover circuits**: "Find circuits for mathematical reasoning"
4. **Visualize results**: "Show me a circuit diagram"
5. **Analyze attention**: "Analyze attention patterns in layer 8"
6. **Generate report**: "Create a comprehensive analysis report"

## 🔧 MLX Engine Background Service

The MCP server requires a persistent MLX Engine API service to handle analysis requests. A dedicated background service is provided for this purpose.

### Service Management Commands

```bash
# Start the service (must be run from mcp-server directory)
./start_mlx_service.sh start

# Check service status and health
./start_mlx_service.sh status

# View recent service logs
./start_mlx_service.sh logs

# Stop the service
./start_mlx_service.sh stop

# Restart the service
./start_mlx_service.sh restart
```

### What the Service Provides

- ✅ **MLX Engine API Server** on port 50111
- ✅ **Model Loading** and generation capabilities
- ✅ **Activation Capture** for mechanistic interpretability
- ✅ **Persistent Background Operation** 
- ✅ **Comprehensive Logging** to `mlx_engine_service.log`
- ✅ **Health Monitoring** and status reporting

### Service Requirements

- **Python 3.9+** with MLX Engine dependencies installed
- **Model Files** in `../models/nightmedia/gpt-oss-20b-q4-hi-mlx/`
- **Sufficient Memory** for model loading (32GB+ recommended)
- **Port 50111** available for the API server

The service runs independently and provides the backend functionality that the MCP server tools depend on.

## 🔍 Troubleshooting

### Common Issues

**Anthropic API Key Error**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Visualization Not Opening**
- Check if port 8888 is available
- Try manually visiting http://localhost:8888
- Use `viz` command to restart visualization server

**MLX Engine Connection Failed**
- Start the background service: `./start_mlx_service.sh start`
- Check service status: `./start_mlx_service.sh status`
- View service logs: `./start_mlx_service.sh logs`
- Verify the MLX Engine REST API is accessible at http://localhost:50111
- Check MLX_ENGINE_API_URL in your environment if using custom URL

**MLX Engine Service Issues**
- Check Python 3 is available: `python3 --version`
- Verify MLX Engine dependencies: `python3 -c "import mlx_engine"`
- Ensure model files exist: `ls ../models/nightmedia/gpt-oss-20b-q4-hi-mlx/`
- Check available memory (32GB+ recommended for full model)
- Monitor service logs: `tail -f mlx_engine_service.log`

### Debug Mode
Set `LOG_LEVEL=DEBUG` for detailed logging:
```bash
LOG_LEVEL=DEBUG npm run chat
```

## 🎯 Next Steps

This implementation provides the foundation for:
1. **Real MLX Engine Integration** - Connect to actual model analysis
2. **Advanced Circuit Discovery** - Implement more sophisticated algorithms
3. **NeuroScope Bridge** - Full integration with Smalltalk interface
4. **Multi-Model Support** - Analyze different model architectures
5. **Collaborative Features** - Share analyses and visualizations

## 📚 Learn More

- [MLX Engine Documentation](../README.md)
- [Visualization System Guide](src/visualization/README.md)
- [MCP Protocol Specification](https://modelcontextprotocol.io/)
- [Mechanistic Interpretability Resources](https://www.anthropic.com/research)

## 🤝 Contributing

This is part of the larger MLX Engine with NeuroScope Integration project. See the main repository for contribution guidelines.

---

**Ready to explore the inner workings of neural networks? Start the chatbot and begin your journey into mechanistic interpretability!**

```bash
npm run chat
```