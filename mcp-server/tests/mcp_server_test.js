/**
 * MCP Server tests
 */

import { assertEquals, assertExists } from '@std/assert';
import { MCPServer } from '../src/server/mcp_server.js';
import { ToolRegistry } from '../src/server/tool_registry.js';
import { RequestRouter } from '../src/server/request_router.js';
import { coreTools } from '../src/services/core_tools.js';

Deno.test('ToolRegistry - register and execute tools', async () => {
  const registry = new ToolRegistry();
  
  // Register core tools
  registry.registerTools(coreTools);
  
  // Test tool registration
  assertEquals(registry.getAllTools().length, 3);
  assertExists(registry.getTool('ping'));
  assertExists(registry.getTool('core_discover_circuits'));
  
  // Test tool execution
  const pingResult = await registry.executeTool('ping', { message: 'test' });
  assertEquals(pingResult.success, true);
  assertEquals(pingResult.message, 'pong');
  
  // Test tool with parameters
  const circuitResult = await registry.executeTool('core_discover_circuits', {
    phenomenon: 'IOI',
    model_id: 'test_model',
  });
  assertEquals(circuitResult.success, true);
  assertEquals(circuitResult.circuits.length, 1);
});

Deno.test('RequestRouter - route MCP requests', async () => {
  const registry = new ToolRegistry();
  registry.registerTools(coreTools);
  
  const router = new RequestRouter(registry);
  
  // Test tools/list request
  const listRequest = {
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/list',
  };
  
  const listResponse = await router.routeRequest(listRequest);
  assertEquals(listResponse.jsonrpc, '2.0');
  assertEquals(listResponse.id, 1);
  assertExists(listResponse.result);
  assertEquals(listResponse.result.tools.length, 3);
  
  // Test tools/call request
  const callRequest = {
    jsonrpc: '2.0',
    id: 2,
    method: 'tools/call',
    params: {
      name: 'ping',
      arguments: { message: 'test' },
    },
  };
  
  const callResponse = await router.routeRequest(callRequest);
  assertEquals(callResponse.jsonrpc, '2.0');
  assertEquals(callResponse.id, 2);
  assertExists(callResponse.result);
  assertExists(callResponse.result.content);
});

Deno.test('MCPServer - initialization and tool management', async () => {
  const config = {
    mcp: { port: 3000, host: 'localhost' },
    mlxEngine: { apiUrl: 'http://localhost:50111' },
    storage: { activationsPath: './data/activations' },
    analysis: { defaultTimeout: 60000 },
    security: { enableAuth: false },
    logging: { level: 'INFO' },
  };
  
  const server = new MCPServer(config);
  
  // Test tool registration
  server.registerTools(coreTools);
  
  const stats = server.getStats();
  assertEquals(stats.totalTools, 3);
  assertEquals(stats.toolCategories.core, 2);
  assertEquals(stats.toolCategories.ping, 1);
  
  // Test tool retrieval
  const pingTool = server.getTool('ping');
  assertExists(pingTool);
  assertEquals(pingTool.name, 'ping');
  
  const coreToolsList = server.getToolsByCategory('core');
  assertEquals(coreToolsList.length, 2);
});

Deno.test('Error handling - invalid requests', async () => {
  const registry = new ToolRegistry();
  const router = new RequestRouter(registry);
  
  // Test invalid JSON-RPC request
  const invalidRequest = {
    id: 1,
    method: 'invalid_method',
  };
  
  const response = await router.routeRequest(invalidRequest);
  assertExists(response.error);
  assertEquals(response.error.code, -32600); // INVALID_REQUEST
  
  // Test method not found
  const notFoundRequest = {
    jsonrpc: '2.0',
    id: 2,
    method: 'nonexistent_method',
  };
  
  const notFoundResponse = await router.routeRequest(notFoundRequest);
  assertExists(notFoundResponse.error);
  assertEquals(notFoundResponse.error.code, -32601); // METHOD_NOT_FOUND
});

Deno.test('Schema validation - tool parameters', async () => {
  const registry = new ToolRegistry();
  registry.registerTools(coreTools);
  
  // Test valid parameters
  const validResult = await registry.executeTool('core_discover_circuits', {
    phenomenon: 'IOI',
    model_id: 'test_model',
  });
  assertEquals(validResult.success, true);
  
  // Test invalid parameters - should throw error
  try {
    await registry.executeTool('core_discover_circuits', {
      phenomenon: 'invalid_phenomenon', // Not in enum
      model_id: 'test_model',
    });
    throw new Error('Should have thrown validation error');
  } catch (error) {
    assertEquals(error.message.includes('validation'), true);
  }
});