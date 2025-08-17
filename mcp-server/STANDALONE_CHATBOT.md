# Standalone Neural Circuit Chatbot

## Overview

The standalone chatbot provides a command-line interface to the same neural circuit analysis tools that Cursor uses through the MCP server. It offers real-time GPT-OSS-20B analysis with interactive WebGL2 visualizations.

## Quick Start

```bash
# Start the chatbot
npm run chat

# Run demo session
npm run demo

# Test functionality
npm run chat:test
```

## Features

### ✅ Real MCP Server Integration
- **Same tools as Cursor**: Uses identical implementations from `filesystem_pattern_server.js`
- **Version verification**: Ensures MCP server version 25+ on startup
- **Real API calls**: Direct HTTP requests to MLX Engine (port 50111) and Visualization server (port 8888)

### ✅ Startup Health Checks
- **MCP server version check**: Verifies server is running and up-to-date
- **Service health monitoring**: Checks MLX Engine and Visualization server status
- **Graceful error handling**: Continues with warnings if services are unavailable

### ✅ Demo Documentation Integration
- **Full knowledge base**: Loads `NEURAL_CIRCUIT_VISUALIZATION_DEMO.md` into system prompt
- **17,000+ character context**: Complete guide for neural circuit analysis workflows
- **Conversational expertise**: Can answer questions about system architecture and usage

### ✅ Interactive Commands
```
help      - Show command reference
status    - System status with health data  
health    - Comprehensive service health check
demo      - Complete neural circuit visualization demo
tools     - List available analysis tools
history   - Show conversation history  
clear     - Clear screen and show welcome
viz       - Open visualization interface
exit/quit - Exit the chatbot
```

### ✅ Automatic Workflows
- **Math problem generation**: Creates varied arithmetic problems for circuit analysis
- **Activation capture**: Real neural activations from GPT-OSS-20B during inference
- **Circuit visualization**: Converts raw activations to WebGL2 interactive graphs
- **Browser integration**: Automatically opens visualization in browser

## Example Session

```
🧠 MI-Chat> demo

🎬 Neural Circuit Visualization Demo
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

This demo will:
1. Check system health
2. Load GPT-OSS-20B model
3. Capture neural activations for a math problem
4. Create an interactive WebGL2 visualization
5. Open the results in your browser

🔍 Step 1: Checking system health...
✅ MCP server running - Version 25
✅ MLX Engine: 🟢 healthy
✅ Visualization: 🟢 healthy

Press Enter to continue...
```

## Technical Architecture

### Real MCP Tools Implementation
The chatbot creates exact replicas of the MCP server tools:

```javascript
// Example: load_model tool
createLoadModelTool() {
  return {
    name: 'load_model',
    handler: async (args) => {
      const response = await fetch('http://localhost:50111/v1/models/load', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: `/Users/craig/me/behavior/forks/mlx-engine-neuroscope/models/nightmedia/${args.model_id}`,
          model_id: args.model_id
        })
      });
      return await response.json();
    }
  };
}
```

### System Prompt Enhancement
```javascript
// Loads complete demo documentation
const demoPath = path.join(__dirname, '../../NEURAL_CIRCUIT_VISUALIZATION_DEMO.md');
const demoDocumentation = await fs.promises.readFile(demoPath, 'utf8');

// 17,000+ character system prompt including:
// - Available MCP tools
// - System architecture details  
// - Demo workflows and examples
// - Troubleshooting guides
// - Expected outputs and results
```

### Health Monitoring
```javascript
// Startup checks
await this.ensureMCPServerRunning();  // Version verification
await this.checkAllServicesHealth(); // MLX + Viz server status

// Runtime health commands
health_check: { service: 'all' }      // Comprehensive health report
start_server: { service: 'mlx' }      // Auto-start services
```

## Available NPM Scripts

```json
{
  "chat": "node src/cli/interactive_chat.js",
  "chat:test": "node test_interactive_chat.js", 
  "demo": "echo '🎬 Starting Neural Circuit Demo...' && npm run chat",
  "viz": "node src/visualization/server.js"
}
```

## File Structure

```
mcp-server/
├── src/cli/interactive_chat.js           # Main chatbot implementation
├── test_interactive_chat.js              # Test suite
├── start_chatbot.sh                      # Bash startup script  
├── NEURAL_CIRCUIT_VISUALIZATION_DEMO.md  # Demo documentation
└── package.json                          # NPM scripts
```

## Requirements

1. **Node.js 18+** - Runtime environment
2. **Anthropic API key** - For conversational AI (pre-configured)
3. **MLX Engine** - Running on port 50111 (`./start_mlx_service.sh`)
4. **Visualization server** - Running on port 8888 (`npm run viz`)

## Success Criteria

✅ **MCP Server Integration**: Uses same tools as Cursor (version 25+)  
✅ **Real Data Flow**: Actual GPT-OSS-20B activations, never mock data  
✅ **WebGL2 Visualization**: Hardware-accelerated Cosmos Graph rendering  
✅ **Health Monitoring**: Comprehensive service status and auto-startup  
✅ **Demo Documentation**: Complete 17,000+ char knowledge base loaded  
✅ **Interactive Experience**: Natural language commands with tool execution  

## Comparison: Standalone vs Cursor

| Feature | Standalone Chatbot | Cursor + MCP |
|---------|-------------------|--------------|
| **Tools** | Same exact implementations | ✅ Real MCP server |
| **Data** | Real GPT-OSS-20B activations | ✅ Real GPT-OSS-20B activations |
| **Visualization** | WebGL2 Cosmos Graph | ✅ WebGL2 Cosmos Graph |
| **Interface** | Command-line chat | ✅ IDE integration |
| **Health Checks** | Comprehensive monitoring | ✅ Comprehensive monitoring |
| **Demo Knowledge** | Full documentation loaded | ✅ Full documentation loaded |

The standalone chatbot provides **identical functionality** to the Cursor integration, with the convenience of a dedicated command-line interface for neural circuit analysis.
