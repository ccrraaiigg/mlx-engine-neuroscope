#!/usr/bin/env node

/**
 * Interactive CLI Chatbot for Mechanistic Interpretability MCP Server
 * 
 * This chatbot provides a natural language interface to the MCP server,
 * allowing users to perform mechanistic interpretability operations through
 * conversation and automatically display visualizations.
 */

import readline from 'readline';
import { Anthropic } from '@anthropic-ai/sdk';
import { MCPServer } from '../server/mcp_server.js';
import { loadConfig } from '../config/config.js';
import { setupLogging, getLogger } from '../utils/logging.js';
import { coreTools } from '../services/core_tools.js';
import { mlxTools, initializeMLXClient, getMLXClient } from '../services/mlx_tools.js';
import { vizTools } from '../services/viz_tools.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

class InteractiveChatbot {
  constructor(config) {
    this.config = config;
    this.logger = getLogger('InteractiveChatbot');
    this.conversationHistory = [];
    this.sessionId = `session_${Date.now()}`;
    
    // Initialize Anthropic client
    this.anthropic = new Anthropic({
      apiKey: config.anthropic.apiKey,
    });
    
    // Initialize MCP server
    this.mcpServer = new MCPServer(config);
    
    // Initialize MLX Engine client (real connection)
    initializeMLXClient(config.mlxEngine, false);
    
    // Register all tools
    this.mcpServer.registerTools(coreTools);
    this.mcpServer.registerTools(mlxTools);
    this.mcpServer.registerTools(vizTools);
    
    // Initialize readline interface
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: '🧠 MI-Chat> ',
    });
    
    this.setupEventHandlers();
  }

  setupEventHandlers() {
    this.rl.on('line', (input) => this.handleUserInput(input.trim()));
    this.rl.on('close', () => this.shutdown());
    
    // Handle Ctrl+C gracefully
    process.on('SIGINT', () => {
      console.log('\n👋 Goodbye! Thanks for exploring mechanistic interpretability!');
      this.shutdown();
    });
  }

  async start() {
    try {
      this.logger.info('Starting Interactive Chatbot...');
      
      // Display welcome message
      this.displayWelcome();
      
      // Start the conversation loop
      this.rl.prompt();
      
    } catch (error) {
      this.logger.error(`Failed to start chatbot: ${error.message}`);
      console.error('❌ Failed to start chatbot:', error.message);
      process.exit(1);
    }
  }

  displayWelcome() {
    console.log(`
╔══════════════════════════════════════════════════════════════════════════════╗
║                    🧠 Mechanistic Interpretability Chatbot                   ║
║                                                                              ║
║  Welcome! I can help you analyze neural networks and discover circuits.     ║
║  I have access to MLX Engine and can show you beautiful visualizations.     ║
║                                                                              ║
║  Try asking me:                                                              ║
║  • "Show me circuits in GPT-OSS-20B for indirect object identification"     ║
║  • "Visualize attention patterns in layer 8"                                ║
║  • "Find neurons that activate for mathematical reasoning"                   ║
║  • "help" - for more commands                                                ║
║                                                                              ║
║  Session ID: ${this.sessionId}                                    ║
╚══════════════════════════════════════════════════════════════════════════════╝
`);
  }

  async handleUserInput(input) {
    if (!input) {
      this.rl.prompt();
      return;
    }

    // Handle special commands
    if (await this.handleSpecialCommands(input)) {
      this.rl.prompt();
      return;
    }

    try {
      console.log('🤔 Thinking...');
      
      // Add user message to conversation history
      this.conversationHistory.push({
        role: 'user',
        content: input,
        timestamp: new Date().toISOString(),
      });

      // Get response from Anthropic
      const response = await this.getAnthropicResponse(input);
      
      // Add assistant response to history
      this.conversationHistory.push({
        role: 'assistant',
        content: response,
        timestamp: new Date().toISOString(),
      });

      // Display response
      console.log(`\n🤖 ${response}\n`);
      
      // Check if we need to execute any tools
      await this.executeToolsFromResponse(response);
      
    } catch (error) {
      this.logger.error(`Error handling user input: ${error.message}`);
      console.error('❌ Sorry, I encountered an error:', error.message);
    }

    this.rl.prompt();
  }

  async handleSpecialCommands(input) {
    const command = input.toLowerCase();
    
    switch (command) {
      case 'help':
        this.displayHelp();
        return true;
        
      case 'status':
        await this.displayStatus();
        return true;
        
      case 'tools':
        this.displayAvailableTools();
        return true;
        
      case 'history':
        this.displayConversationHistory();
        return true;
        
      case 'clear':
        console.clear();
        this.displayWelcome();
        return true;
        
      case 'viz':
      case 'visualize':
        await this.openVisualization();
        return true;
        
      case 'exit':
      case 'quit':
        this.shutdown();
        return true;
        
      default:
        return false;
    }
  }

  displayHelp() {
    console.log(`
📚 Available Commands:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 System Commands:
  help      - Show this help message
  status    - Show system status and configuration
  tools     - List available analysis tools
  history   - Show conversation history
  clear     - Clear screen and show welcome message
  viz       - Open visualization interface in browser
  exit/quit - Exit the chatbot

🧠 Analysis Examples:
  "Discover circuits for indirect object identification in GPT-OSS-20B"
  "Show me attention patterns in layer 8 for the phrase 'hello world'"
  "Find neurons that activate strongly for mathematical operations"
  "Visualize the circuit that handles factual recall"
  "Analyze feature entanglement between language and reasoning"

🎯 Visualization Examples:
  "Show me a graph of the IOI circuit"
  "Visualize attention flow for this sentence"
  "Create a circuit diagram for mathematical reasoning"
  "Display activation patterns across all layers"

💡 Tips:
  • Be specific about what you want to analyze
  • Mention model names (like GPT-OSS-20B) when relevant
  • Ask for visualizations to see your results graphically
  • Use natural language - I'll translate to the right tools!

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
  }

  async displayStatus() {
    const stats = this.mcpServer.getStats();
    const config = this.config;
    
    console.log(`
📊 System Status:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

🔧 MCP Server:
  • Tools Registered: ${stats.totalTools}
  • Tool Categories: ${Object.keys(stats.toolCategories).join(', ')}
  • Session ID: ${this.sessionId}

🌐 MLX Engine:
  • API URL: ${config.mlxEngine?.apiUrl || 'Not configured'}
  • Status: ${await this.checkMLXEngineStatus()}

💬 Conversation:
  • Messages: ${this.conversationHistory.length}
  • Started: ${new Date(parseInt(this.sessionId.split('_')[1])).toLocaleString()}

🎨 Visualization:
  • Available: ✅ Cosmos Graph WebGL
  • Fallbacks: Canvas 2D, SVG
  • Server: http://localhost:3000

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);
  }

  displayAvailableTools() {
    const tools = this.mcpServer.getAllTools();
    
    console.log(`
🛠️  Available Analysis Tools:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);

    const categories = {};
    tools.forEach(tool => {
      const category = tool.name.split('_')[0];
      if (!categories[category]) categories[category] = [];
      categories[category].push(tool);
    });

    Object.entries(categories).forEach(([category, categoryTools]) => {
      console.log(`\n📂 ${category.toUpperCase()} Tools:`);
      categoryTools.forEach(tool => {
        console.log(`  • ${tool.name} - ${tool.description}`);
      });
    });

    console.log(`
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
💡 Just ask me in natural language and I'll use the right tools!
`);
  }

  displayConversationHistory() {
    console.log(`
💬 Conversation History (${this.conversationHistory.length} messages):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
`);

    this.conversationHistory.forEach((message, index) => {
      const time = new Date(message.timestamp).toLocaleTimeString();
      const icon = message.role === 'user' ? '👤' : '🤖';
      const content = message.content.length > 100 
        ? message.content.substring(0, 100) + '...'
        : message.content;
      
      console.log(`${index + 1}. [${time}] ${icon} ${content}`);
    });

    console.log('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━');
  }

  async openVisualization() {
    try {
      console.log('🎨 Opening visualization interface...');
      
      // Start the visualization server
      const vizServerPath = path.join(__dirname, '../../visualization/server.js');
      const vizProcess = spawn('node', [vizServerPath], {
        detached: true,
        stdio: 'ignore'
      });
      
      vizProcess.unref();
      
      console.log(`
🌐 Visualization server starting...
   URL: http://localhost:8888
   
   The visualization interface will open in your browser.
   You can view circuit diagrams, attention patterns, and more!
`);
      
      // Try to open browser (platform-specific)
      const platform = process.platform;
      let openCommand;
      
      if (platform === 'darwin') {
        openCommand = 'open';
      } else if (platform === 'win32') {
        openCommand = 'start';
      } else {
        openCommand = 'xdg-open';
      }
      
      spawn(openCommand, ['http://localhost:8888'], {
        detached: true,
        stdio: 'ignore'
      }).unref();
      
    } catch (error) {
      console.error('❌ Failed to open visualization:', error.message);
    }
  }

  async getAnthropicResponse(userInput) {
    const systemPrompt = this.buildSystemPrompt();
    
    const messages = [
      { role: 'user', content: userInput }
    ];
    
    // Add recent conversation context (last 10 messages)
    const recentHistory = this.conversationHistory.slice(-10);
    recentHistory.forEach(msg => {
      messages.unshift({
        role: msg.role,
        content: msg.content
      });
    });

    const response = await this.anthropic.messages.create({
      model: 'claude-3-sonnet-20240229',
      max_tokens: 1000,
      system: systemPrompt,
      messages: messages
    });

    return response.content[0].text;
  }

  buildSystemPrompt() {
    const availableTools = this.mcpServer.getAllTools();
    const toolDescriptions = availableTools.map(tool => 
      `- ${tool.name}: ${tool.description}`
    ).join('\n');

    return `You are an expert assistant for mechanistic interpretability research. You help users analyze neural networks, discover circuits, and understand model behavior.

Available MCP Tools:
${toolDescriptions}

Your capabilities:
1. Interpret natural language requests about neural network analysis
2. Suggest appropriate tools and analyses
3. Explain mechanistic interpretability concepts clearly
4. Guide users through complex analysis workflows
5. Recommend visualizations for better understanding

When users ask for analysis:
1. Acknowledge their request
2. Explain what analysis would be helpful
3. Mention if you would use specific tools (but don't actually call them - the system will handle that)
4. Suggest visualizations when appropriate
5. Provide educational context about the concepts involved

Be conversational, helpful, and educational. Focus on making mechanistic interpretability accessible and engaging.

Current session context:
- User is interacting through a CLI chatbot
- Visualization system is available at http://localhost:8888
- MLX Engine integration provides real model data
- Focus on practical, actionable insights`;
  }

  async executeToolsFromResponse(response) {
    // Enhanced pattern matching for different types of analysis
    const toolPatterns = [
      // Core analysis tools
      { pattern: /discover.*circuit/i, tool: 'core_discover_circuits', params: { phenomenon: 'IOI', model_id: 'gpt-oss-20b' }},
      { pattern: /localize.*feature/i, tool: 'core_localize_features', params: { feature_name: 'mathematical_reasoning', model_id: 'gpt-oss-20b' }},
      
      // MLX Engine tools
      { pattern: /load.*model|model.*load/i, tool: 'mlx_load_model', params: { model_id: 'gpt-oss-20b' }},
      { pattern: /math.*analysis|analyze.*math/i, tool: 'mlx_analyze_math', params: { prompt: '2 + 2 = ?' }},
      { pattern: /attention.*pattern|analyze.*attention/i, tool: 'mlx_analyze_attention', params: { prompt: 'The cat sat on the mat', layers: [8, 10, 12] }},
      { pattern: /factual.*recall|analyze.*fact/i, tool: 'mlx_analyze_factual', params: { query: 'What is the capital of France?' }},
      { pattern: /capture.*activation/i, tool: 'mlx_capture_activations', params: { prompt: 'Hello world' }},
      
      // Visualization tools
      { pattern: /visualize.*circuit|circuit.*diagram/i, tool: 'viz_circuit_diagram', params: { circuit_data: { components: ['attention_head_8_3', 'mlp_9', 'attention_head_10_1'], confidence: 0.85 }, circuit_name: 'Demo Circuit' }},
      { pattern: /visualize.*attention|attention.*viz/i, tool: 'viz_attention_patterns', params: { attention_data: { patterns: [{ layer: 8, head: 3, pattern_type: 'induction', strength: 0.85, tokens_involved: ['the', 'cat', 'sat'] }] }, layers: [8] }},
      { pattern: /open.*browser|browser.*viz/i, tool: 'viz_open_browser', params: {}},
      
      // Test tools
      { pattern: /ping|test/i, tool: 'ping', params: { message: 'test from chatbot' }}
    ];

    for (const { pattern, tool, params } of toolPatterns) {
      if (pattern.test(response)) {
        try {
          console.log(`🔧 Executing ${tool}...`);
          const result = await this.mcpServer.getTool(tool).handler(params);
          
          console.log(`✅ Tool execution completed:`);
          console.log(JSON.stringify(result, null, 2));
          
          // Auto-open visualization for certain results
          if (result.success && result.visualization_url) {
            console.log(`\n🎨 Opening visualization: ${result.visualization_url}`);
            await this.mcpServer.getTool('viz_open_browser').handler({ url: result.visualization_url });
          }
          
          // Suggest visualization for analysis results
          if (tool.includes('analyze') || tool.includes('discover') || tool.includes('localize')) {
            console.log(`\n💡 Tip: I can create visualizations of this data! Try asking me to "visualize the results"`);
          }
          
        } catch (error) {
          console.error(`❌ Tool execution failed: ${error.message}`);
        }
        break; // Only execute the first matching tool
      }
    }
  }

  async checkMLXEngineStatus() {
    try {
      const client = getMLXClient();
      await client.checkHealth();
      return '🟢 Connected';
    } catch (error) {
      return `🔴 Connection failed: ${error.message}`;
    }
  }

  shutdown() {
    console.log('\n📊 Session Summary:');
    console.log(`   Messages exchanged: ${this.conversationHistory.length}`);
    console.log(`   Session duration: ${Math.round((Date.now() - parseInt(this.sessionId.split('_')[1])) / 1000)}s`);
    console.log('\n🔬 Keep exploring the mysteries of neural networks!');
    
    this.rl.close();
    process.exit(0);
  }
}

// Main execution
async function main() {
  try {
    // Load configuration
    const config = await loadConfig();
    
    // Setup logging
    setupLogging(config.logging);
    
    // Check for Anthropic API key
    if (!config.anthropic?.apiKey) {
      console.error(`
❌ Anthropic API key not found!

Please add your API key to the keys file:
  echo "anthropic=your-api-key-here" >> keys

Or set environment variable:
  export ANTHROPIC_API_KEY="your-api-key-here"

Get your API key at: https://console.anthropic.com/
`);
      process.exit(1);
    }
    
    // Create and start chatbot
    const chatbot = new InteractiveChatbot(config);
    await chatbot.start();
    
  } catch (error) {
    console.error('❌ Failed to start chatbot:', error.message);
    process.exit(1);
  }
}

// Run if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}

export { InteractiveChatbot };