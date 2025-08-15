# Mechanistic Interpretability MCP Server

A Model Context Protocol (MCP) server that provides comprehensive mechanistic interpretability capabilities for Large Language Models, with specialized integration for MLX Engine on Apple Silicon.

## Features

- **Circuit Analysis**: Discover and analyze computational circuits within transformer models
- **Model Modification**: Apply targeted interventions and modifications to model behavior
- **Safety Validation**: Comprehensive safety testing and validation capabilities
- **MLX Integration**: Optimized for Apple Silicon using MLX framework
- **Real-time Analysis**: Stream activations and perform live analysis during inference

## Quick Start

```bash
# Install Node.js (if not already installed)
# Visit https://nodejs.org/ or use a package manager like homebrew:
# brew install node

# Install dependencies
npm install

# Run the server
npm start

# Or run in development mode with auto-reload
npm run dev

# Test the server functionality
node src/test_server.js
```

## Interactive Chatbot

The MCP server includes an interactive command-line chatbot that uses the Anthropic API to provide natural language access to mechanistic interpretability operations.

### Setup

1. **API Keys**: Ensure you have API keys configured in the `keys` file:
   ```
   anthropic=sk-ant-api03-your-key-here
   openai=sk-proj-your-key-here
   ```

2. **Start the chatbot**:
   ```bash
   # Run the interactive chatbot
   npm run chatbot
   
   # Or run directly
   node src/chatbot.js
   ```

### Usage

The chatbot provides a conversational interface to all MCP server capabilities:

```
🧠 Mechanistic Interpretability Chatbot
Connected to MCP Server with 15 tools available.

> Hello! Can you help me discover circuits for indirect object identification?

I'll help you discover circuits for indirect object identification (IOI). This is a well-studied 
phenomenon in mechanistic interpretability. Let me run the circuit discovery tool for you.

[Executing: core_discover_circuits with phenomenon="IOI"]

✅ Found 3 circuit candidates for IOI:
- IOI_primary_circuit (confidence: 0.87)
  - Layers: [8, 9, 10, 11]  
  - Components: attention_head_8_3, mlp_9, attention_head_10_1, attention_head_11_2
  - Performance recovery: 92%

Would you like me to analyze any of these circuits in more detail?

> Yes, can you localize the features in the primary circuit?

I'll localize the specific features within the IOI primary circuit...

[Executing: core_localize_features with circuit_id="IOI_primary_circuit"]
```

### Available Commands

The chatbot understands natural language requests for:

- **Circuit Discovery**: "Find circuits for [phenomenon]", "Discover IOI circuits"
- **Feature Localization**: "Localize features in [circuit]", "Which neurons are responsible for [feature]?"
- **Model Analysis**: "Analyze attention patterns", "Show me the residual stream"
- **Safety Operations**: "Check for harmful circuits", "Validate this modification"
- **Data Management**: "Export results to JSON", "Show me the activation data"

### Help and Examples

```bash
# Get help within the chatbot
> help

# Ask about available capabilities  
> What can you do?

# Request examples
> Show me examples of circuit discovery

# Exit the chatbot
> exit
```

### Features

- **Natural Language Interface**: Ask questions in plain English
- **Automatic Tool Selection**: The chatbot automatically chooses the right MCP tools
- **Conversation History**: Maintains context throughout your session
- **Rich Output Formatting**: Results are presented in an easy-to-read format
- **Error Explanation**: Errors are explained in plain language with suggestions
- **Session Management**: Save and restore conversation sessions

### Configuration

Configure the chatbot behavior via environment variables:

```bash
# Anthropic API settings
export ANTHROPIC_API_KEY=your-key-here
export ANTHROPIC_MODEL=claude-3-sonnet-20240229

# Chatbot settings
export CHATBOT_MAX_HISTORY=50
export CHATBOT_SAVE_SESSIONS=true
export CHATBOT_SESSION_DIR=./sessions

# Start chatbot with custom settings
npm run chatbot
```

## Current Implementation Status

✅ **Task 1.2 Complete**: MCP server core framework implemented
- MCP SDK integration with @modelcontextprotocol/sdk
- Tool registry with comprehensive JSON schema validation
- Request routing system with proper error handling
- Logging system using Node.js built-in modules
- Basic core tools for testing (ping, core_discover_circuits, core_localize_features)

🚧 **Next Steps**: Additional tool implementations as defined in tasks.md

## Configuration

### API Keys Setup

Create a `keys` file in the mcp-server directory with your API keys:

```
anthropic=sk-ant-api03-your-anthropic-key-here
openai=sk-proj-your-openai-key-here
```

**Note**: The `keys` file is automatically ignored by git for security.

### Server Configuration

The server can be configured via environment variables or a `config.json` file:

```bash
# Environment variables
export MCP_PORT=3000
export MCP_HOST=localhost
export MLX_ENGINE_API_URL=http://localhost:8080
export LOG_LEVEL=INFO

# Start server
npm start
```

### Configuration File

Copy `config.example.json` to `config.json` and modify as needed:

```json
{
  "mcp": {
    "port": 3000,
    "host": "localhost"
  },
  "mlxEngine": {
    "apiUrl": "http://localhost:8080"
  },
  "logging": {
    "level": "INFO"
  }
}
```

## Development

```bash
# Run tests
npm test

# Format code
npm run format

# Lint code
npm run lint
```

## Troubleshooting

### Chatbot Issues

**"Anthropic API key not found"**
- Ensure the `keys` file exists in the mcp-server directory
- Verify the format: `anthropic=sk-ant-api03-your-key-here`
- Check file permissions are readable

**"Connection refused to MCP server"**
- Start the MCP server first: `npm start`
- Verify the server is running on the correct port
- Check firewall settings

**"Tool execution failed"**
- Ensure MLX Engine is running (if using MLX tools)
- Check the server logs for detailed error messages
- Verify tool parameters are valid

**"Rate limit exceeded"**
- The Anthropic API has rate limits
- Wait a moment before making more requests
- Consider upgrading your API plan for higher limits

### General Issues

**"Permission denied"**
- Ensure Node.js has the required permissions for file system and network access
- Check file system permissions for data directories

**"Module not found"**
- Run `npm install` to ensure dependencies are installed
- Check your internet connection for downloading dependencies

## Requirements

- Node.js 18+
- MLX Engine API server running (for full functionality)
- macOS with Apple Silicon (recommended for MLX integration)

## License

MIT