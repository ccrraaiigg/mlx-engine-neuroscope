/**
 * MCP Server implementation
 * Main server class that implements the Model Context Protocol
 */

import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { CallToolRequestSchema, ListToolsRequestSchema } from '@modelcontextprotocol/sdk/types.js';
import { ToolRegistry } from './tool_registry.js';
import { RequestRouter } from './request_router.js';
import { getLogger } from '../utils/logging.js';

export class MCPServer {
  constructor(config) {
    this.config = config;
    this.logger = getLogger('MCPServer');
    this.running = false;

    // Initialize core components
    this.toolRegistry = new ToolRegistry();
    this.requestRouter = new RequestRouter(this.toolRegistry);

    // Initialize MCP SDK server
    this.server = new Server(
      {
        name: 'mechanistic-interpretability-mcp-server',
        version: '0.1.0',
      },
      {
        capabilities: {
          tools: {},
        },
      },
    );

    this.setupServerHandlers();
  }

  /**
   * Sets up MCP server request handlers
   */
  setupServerHandlers() {
    // Handle tool listing
    this.server.setRequestHandler(ListToolsRequestSchema, () => {
      const tools = this.toolRegistry.getAllTools();
      return {
        tools: tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        })),
      };
    });

    // Handle tool calls
    this.server.setRequestHandler(CallToolRequestSchema, async (request) => {
      try {
        const { name, arguments: args } = request.params;

        this.logger.info(`Executing tool: ${name}`);

        const result = await this.toolRegistry.executeTool(name, args || {});

        return {
          content: [
            {
              type: 'text',
              text: typeof result === 'string' ? result : JSON.stringify(result, null, 2),
            },
          ],
        };
      } catch (error) {
        this.logger.error(`Tool execution failed: ${error.message}`);

        return {
          content: [
            {
              type: 'text',
              text: `Error: ${error.message}`,
            },
          ],
          isError: true,
        };
      }
    });
  }

  /**
   * Registers a tool with the server
   * @param {object} tool - Tool definition
   */
  registerTool(tool) {
    this.toolRegistry.registerTool(tool);
  }

  /**
   * Registers multiple tools with the server
   * @param {Array<object>} tools - Array of tool definitions
   */
  registerTools(tools) {
    this.toolRegistry.registerTools(tools);
  }

  /**
   * Starts the MCP server
   */
  async start() {
    try {
      this.logger.info('Starting MCP Server...');

      // Create transport (stdio for MCP)
      const transport = new StdioServerTransport();

      // Connect server to transport
      await this.server.connect(transport);

      this.running = true;
      this.logger.info('MCP Server started successfully');

      // Log server statistics
      const stats = this.getStats();
      this.logger.info(`Server ready with ${stats.totalTools} tools registered`);
    } catch (error) {
      this.logger.error(`Failed to start MCP Server: ${error.message}`);
      throw error;
    }
  }

  /**
   * Stops the MCP server
   */
  async stop() {
    try {
      this.logger.info('Stopping MCP Server...');

      if (this.server) {
        await this.server.close();
      }

      this.running = false;
      this.logger.info('MCP Server stopped successfully');
    } catch (error) {
      this.logger.error(`Error stopping MCP Server: ${error.message}`);
      throw error;
    }
  }

  /**
   * Gets server status
   * @returns {object} Server status information
   */
  getStatus() {
    return {
      running: this.running,
      config: {
        name: this.server?.serverInfo?.name,
        version: this.server?.serverInfo?.version,
      },
      uptime: this.running ? Date.now() - this.startTime : 0,
      stats: this.getStats(),
    };
  }

  /**
   * Gets server statistics
   * @returns {object} Server statistics
   */
  getStats() {
    const toolStats = this.toolRegistry.getStats();
    const routerStats = this.requestRouter.getStats();

    return {
      totalTools: toolStats.totalTools,
      toolCategories: toolStats.categories,
      registeredMethods: routerStats.registeredMethods,
      availableMethods: routerStats.methods,
    };
  }

  /**
   * Validates server configuration
   * @returns {boolean} True if configuration is valid
   * @throws {Error} If configuration is invalid
   */
  validateConfiguration() {
    if (!this.config) {
      throw new Error('Server configuration is required');
    }

    if (!this.config.mcp) {
      throw new Error('MCP configuration is required');
    }

    if (!this.config.mlxEngine) {
      throw new Error('MLX Engine configuration is required');
    }

    if (!this.config.mlxEngine.apiUrl) {
      throw new Error('MLX Engine API URL is required');
    }

    return true;
  }

  /**
   * Handles graceful shutdown
   */
  async gracefulShutdown() {
    this.logger.info('Initiating graceful shutdown...');

    try {
      // Stop accepting new requests
      this.running = false;

      // Wait for ongoing requests to complete (with timeout)
      await this.waitForRequestsToComplete(5000);

      // Close server
      await this.stop();

      this.logger.info('Graceful shutdown completed');
    } catch (error) {
      this.logger.error(`Error during graceful shutdown: ${error.message}`);
      throw error;
    }
  }

  /**
   * Waits for ongoing requests to complete
   * @param {number} timeout - Timeout in milliseconds
   * @returns {Promise<void>}
   */
  waitForRequestsToComplete(timeout = 5000) {
    return new Promise((resolve) => {
      const startTime = Date.now();

      const checkRequests = () => {
        // In a real implementation, you'd track ongoing requests
        // For now, just wait a short time
        if (Date.now() - startTime >= timeout) {
          resolve();
        } else {
          setTimeout(checkRequests, 100);
        }
      };

      checkRequests();
    });
  }

  /**
   * Gets tool by name
   * @param {string} name - Tool name
   * @returns {object|null} Tool definition or null
   */
  getTool(name) {
    return this.toolRegistry.getTool(name);
  }

  /**
   * Gets all registered tools
   * @returns {Array<object>} Array of tool definitions
   */
  getAllTools() {
    return this.toolRegistry.getAllTools();
  }

  /**
   * Gets tools by category
   * @param {string} category - Category name
   * @returns {Array<object>} Array of tools in category
   */
  getToolsByCategory(category) {
    return this.toolRegistry.getToolsByCategory(category);
  }
}
