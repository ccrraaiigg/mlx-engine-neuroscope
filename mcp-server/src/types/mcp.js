/**
 * MCP protocol types and schemas
 */

import { z } from 'zod';

// JSON Schema definition for tool parameters and responses
export const JSONSchemaPropertySchema = z.object({
  type: z.enum(['string', 'number', 'integer', 'boolean', 'array', 'object']),
  description: z.string().optional(),
  enum: z.array(z.any()).optional(),
  minimum: z.number().optional(),
  maximum: z.number().optional(),
  minLength: z.number().optional(),
  maxLength: z.number().optional(),
  pattern: z.string().optional(),
  items: z.lazy(() => JSONSchemaPropertySchema).optional(),
  properties: z.record(z.lazy(() => JSONSchemaPropertySchema)).optional(),
  required: z.array(z.string()).optional(),
  additionalProperties: z.boolean().optional(),
  default: z.any().optional(),
});

export const JSONSchemaSchema = z.object({
  type: z.literal('object'),
  properties: z.record(JSONSchemaPropertySchema),
  required: z.array(z.string()).optional(),
  additionalProperties: z.boolean().default(false),
});

// MCP Tool definition schema
export const MCPToolSchema = z.object({
  name: z.string(),
  description: z.string(),
  inputSchema: JSONSchemaSchema,
  outputSchema: JSONSchemaSchema.optional(),
});

// MCP Request schema
export const MCPRequestSchema = z.object({
  jsonrpc: z.literal('2.0'),
  id: z.union([z.string(), z.number()]),
  method: z.string(),
  params: z.record(z.any()).optional(),
});

// MCP Response schema
export const MCPResponseSchema = z.object({
  jsonrpc: z.literal('2.0'),
  id: z.union([z.string(), z.number()]),
  result: z.any().optional(),
  error: z.object({
    code: z.number(),
    message: z.string(),
    data: z.any().optional(),
  }).optional(),
});

// MCP Error codes
export const MCPErrorCodes = {
  PARSE_ERROR: -32700,
  INVALID_REQUEST: -32600,
  METHOD_NOT_FOUND: -32601,
  INVALID_PARAMS: -32602,
  INTERNAL_ERROR: -32603,
  TOOL_NOT_FOUND: -32001,
  VALIDATION_ERROR: -32002,
  EXECUTION_ERROR: -32003,
};

// Tool execution result
export const MCPToolResultSchema = z.object({
  success: z.boolean(),
  data: z.any().optional(),
  error: z.string().optional(),
  metadata: z.record(z.any()).optional(),
});

/**
 * Validates an MCP request
 * @param {unknown} request - Raw request object
 * @returns {object} Validated request
 */
export function validateMCPRequest(request) {
  return MCPRequestSchema.parse(request);
}

/**
 * Validates an MCP tool definition
 * @param {unknown} tool - Raw tool object
 * @returns {object} Validated tool
 */
export function validateMCPTool(tool) {
  return MCPToolSchema.parse(tool);
}

/**
 * Creates a standardized MCP response
 * @param {string|number} id - Request ID
 * @param {any} result - Response result
 * @param {object} error - Error object if any
 * @returns {object} MCP response
 */
export function createMCPResponse(id, result = null, error = null) {
  const response = {
    jsonrpc: '2.0',
    id,
  };

  if (error) {
    response.error = error;
  } else {
    response.result = result;
  }

  return response;
}

/**
 * Creates a standardized MCP error response
 * @param {string|number} id - Request ID
 * @param {number} code - Error code
 * @param {string} message - Error message
 * @param {any} data - Additional error data
 * @returns {object} MCP error response
 */
export function createMCPError(id, code, message, data = null) {
  return createMCPResponse(id, null, {
    code,
    message,
    data,
  });
}
