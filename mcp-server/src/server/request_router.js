/**
 * Request Router for MCP Server
 * Handles routing and processing of MCP requests
 */

import {
  createMCPError,
  createMCPResponse,
  MCPErrorCodes,
  validateMCPRequest,
} from '../types/mcp.js';
import { getLogger } from '../utils/logging.js';

export class RequestRouter {
  constructor(toolRegistry) {
    this.toolRegistry = toolRegistry;
    this.logger = getLogger('RequestRouter');
    this.requestHandlers = new Map();

    // Register built-in MCP methods
    this.registerBuiltinHandlers();
  }

  /**
   * Registers built-in MCP protocol handlers
   */
  registerBuiltinHandlers() {
    // Tools list method
    this.requestHandlers.set('tools/list', () => {
      const tools = this.toolRegistry.getAllTools();
      return {
        tools: tools.map((tool) => ({
          name: tool.name,
          description: tool.description,
          inputSchema: tool.inputSchema,
        })),
      };
    });

    // Tool call method
    this.requestHandlers.set('tools/call', async (params) => {
      if (!params.name) {
        throw new Error('Tool name is required');
      }

      const result = await this.toolRegistry.executeTool(params.name, params.arguments || {});
      return {
        content: [{
          type: 'text',
          text: typeof result === 'string' ? result : JSON.stringify(result, null, 2),
        }],
        isError: false,
      };
    });

    // Server info method
    this.requestHandlers.set('initialize', (params) => {
      return {
        protocolVersion: '2024-11-05',
        capabilities: {
          tools: {},
        },
        serverInfo: {
          name: 'mechanistic-interpretability-mcp-server',
          version: '0.1.0',
        },
      };
    });

    // Ping method
    this.requestHandlers.set('ping', () => {
      return { status: 'ok', timestamp: new Date().toISOString() };
    });
  }

  /**
   * Registers a custom request handler
   * @param {string} method - Method name
   * @param {Function} handler - Handler function
   */
  registerHandler(method, handler) {
    this.requestHandlers.set(method, handler);
    this.logger.info(`Registered handler for method: ${method}`);
  }

  /**
   * Routes and processes an MCP request
   * @param {object} request - Raw MCP request
   * @returns {Promise<object>} MCP response
   */
  async routeRequest(request) {
    let requestId = null;

    try {
      // Validate request format
      const validatedRequest = validateMCPRequest(request);
      requestId = validatedRequest.id;

      this.logger.info(`Processing request: ${validatedRequest.method} (ID: ${requestId})`);

      // Find handler for the method
      const handler = this.requestHandlers.get(validatedRequest.method);
      if (!handler) {
        return createMCPError(
          requestId,
          MCPErrorCodes.METHOD_NOT_FOUND,
          `Method '${validatedRequest.method}' not found`,
        );
      }

      // Execute handler
      const startTime = Date.now();
      const result = await handler(validatedRequest.params || {});
      const executionTime = Date.now() - startTime;

      this.logger.info(`Request completed: ${validatedRequest.method} (${executionTime}ms)`);

      return createMCPResponse(requestId, result);
    } catch (error) {
      this.logger.error(`Request processing failed: ${error.message}`);

      // Determine error code based on error type
      let errorCode = MCPErrorCodes.INTERNAL_ERROR;
      if (error.name === 'ZodError') {
        errorCode = MCPErrorCodes.INVALID_REQUEST;
      } else if (error.message.includes('not found')) {
        errorCode = MCPErrorCodes.METHOD_NOT_FOUND;
      } else if (error.message.includes('validation')) {
        errorCode = MCPErrorCodes.VALIDATION_ERROR;
      }

      return createMCPError(
        requestId || 'unknown',
        errorCode,
        error.message,
        { stack: error.stack },
      );
    }
  }

  /**
   * Processes multiple requests in batch
   * @param {Array<object>} requests - Array of MCP requests
   * @returns {Promise<Array<object>>} Array of MCP responses
   */
  async routeRequestBatch(requests) {
    if (!Array.isArray(requests)) {
      throw new Error('Batch requests must be an array');
    }

    this.logger.info(`Processing batch of ${requests.length} requests`);

    // Process all requests in parallel
    const responses = await Promise.all(
      requests.map((request) => this.routeRequest(request)),
    );

    return responses;
  }

  /**
   * Gets available methods
   * @returns {Array<string>} Array of available method names
   */
  getAvailableMethods() {
    return Array.from(this.requestHandlers.keys());
  }

  /**
   * Gets router statistics
   * @returns {object} Router statistics
   */
  getStats() {
    return {
      registeredMethods: this.requestHandlers.size,
      methods: this.getAvailableMethods(),
      toolStats: this.toolRegistry.getStats(),
    };
  }

  /**
   * Validates request parameters for a specific method
   * @param {string} method - Method name
   * @param {object} params - Parameters to validate
   * @returns {object} Validated parameters
   * @throws {Error} If validation fails
   */
  validateMethodParams(method, params) {
    // Basic validation for built-in methods
    switch (method) {
      case 'tools/call':
        if (!params.name || typeof params.name !== 'string') {
          throw new Error('Tool name must be a non-empty string');
        }
        if (params.arguments && typeof params.arguments !== 'object') {
          throw new Error('Tool arguments must be an object');
        }
        break;

      case 'initialize':
        if (params.protocolVersion && typeof params.protocolVersion !== 'string') {
          throw new Error('Protocol version must be a string');
        }
        break;
    }

    return params;
  }

  /**
   * Handles errors during request processing
   * @param {Error} error - Error that occurred
   * @param {string} method - Method being processed
   * @param {string|number} requestId - Request ID
   * @returns {object} Error response
   */
  handleError(error, method, requestId) {
    this.logger.error(`Error in ${method}: ${error.message}`, {
      error: error.stack,
      requestId,
      method,
    });

    // Map specific errors to MCP error codes
    if (error.message.includes('Tool') && error.message.includes('not found')) {
      return createMCPError(requestId, MCPErrorCodes.TOOL_NOT_FOUND, error.message);
    }

    if (error.message.includes('validation')) {
      return createMCPError(requestId, MCPErrorCodes.VALIDATION_ERROR, error.message);
    }

    if (error.message.includes('timeout')) {
      return createMCPError(requestId, MCPErrorCodes.INTERNAL_ERROR, 'Request timeout');
    }

    return createMCPError(requestId, MCPErrorCodes.INTERNAL_ERROR, error.message);
  }
}
