/**
 * Simple test script to verify MCP server functionality
 */

import { MCPServer } from './server/mcp_server.js';
import { loadConfig } from './config/config.js';
import { setupLogging } from './utils/logging.js';
import { coreTools } from './services/core_tools.js';

async function testServer() {
  try {
    console.log('Testing MCP Server...');

    // Load configuration
    const config = await loadConfig();
    setupLogging(config.logging);

    // Create server
    const server = new MCPServer(config);

    // Register tools
    server.registerTools(coreTools);

    // Test tool registry
    const stats = server.getStats();
    console.log('Server stats:', stats);

    // Test tool execution
    const pingResult = await server.toolRegistry.executeTool('ping', { message: 'test' });
    console.log('Ping result:', pingResult);

    const circuitResult = await server.toolRegistry.executeTool('core_discover_circuits', {
      phenomenon: 'IOI',
      model_id: 'test_model',
    });
    console.log('Circuit discovery result:', circuitResult);

    console.log('✅ All tests passed!');
  } catch (error) {
    console.error('❌ Test failed:', error.message);
    console.error(error.stack);
    Deno.exit(1);
  }
}

if (import.meta.main) {
  testServer();
}
