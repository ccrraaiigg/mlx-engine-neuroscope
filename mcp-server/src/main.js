/**
 * Main entry point for the Mechanistic Interpretability MCP Server
 *
 * This server provides comprehensive mechanistic interpretability capabilities
 * through the Model Context Protocol (MCP), enabling LLM agents to perform
 * circuit analysis, model modification, and safety validation operations.
 */

import { MCPServer } from './server/mcp_server.js';
import { loadConfig } from './config/config.js';
import { setupLogging } from './utils/logging.js';
import { coreTools } from './services/core_tools.js';

async function main() {
  try {
    // Load configuration
    const config = await loadConfig();

    // Setup logging
    setupLogging(config.logging);

    // Create and start MCP server
    const server = new MCPServer(config);

    // Register core tools
    server.registerTools(coreTools);

    await server.start();

    console.log(`MCP Server started on ${config.mcp.host}:${config.mcp.port}`);
    console.log(`MLX Engine API URL: ${config.mlxEngine.apiUrl}`);

    // Handle graceful shutdown
    const shutdown = () => {
      console.log('Shutting down MCP Server...');
      server.stop();
      Deno.exit(0);
    };

    Deno.addSignalListener('SIGINT', shutdown);
    Deno.addSignalListener('SIGTERM', shutdown);
  } catch (error) {
    console.error('Failed to start MCP Server:', error);
    Deno.exit(1);
  }
}

if (import.meta.main) {
  main();
}
