/**
 * Configuration schemas and validation using Zod
 */

import { z } from 'zod';

// MCP server configuration schema
export const MCPConfigSchema = z.object({
  port: z.number().int().min(1).max(65535).default(3000),
  host: z.string().default('localhost'),
  maxConnections: z.number().int().positive().optional(),
});

// MLX Engine API configuration schema
export const MLXEngineConfigSchema = z.object({
  apiUrl: z.string().url(),
  timeout: z.number().int().positive().default(30000),
  retryAttempts: z.number().int().min(0).default(3),
  apiKey: z.string().optional(),
  maxConcurrentRequests: z.number().int().positive().default(10),
});

// Storage configuration schema
export const StorageConfigSchema = z.object({
  activationsPath: z.string().default('./data/activations'),
  circuitsPath: z.string().default('./data/circuits'),
  cachePath: z.string().default('./data/cache'),
  maxCacheSize: z.number().int().positive().optional(),
});

// Analysis configuration schema
export const AnalysisConfigSchema = z.object({
  defaultTimeout: z.number().int().positive().default(60000),
  cacheResults: z.boolean().default(true),
  maxCircuitCandidates: z.number().int().positive().default(100),
  confidenceThreshold: z.number().min(0).max(1).default(0.8),
});

// Security configuration schema
export const SecurityConfigSchema = z.object({
  enableAuth: z.boolean().default(false),
  apiKeys: z.array(z.string()).optional(),
  rateLimiting: z.object({
    windowMs: z.number().int().positive(),
    maxRequests: z.number().int().positive(),
  }).optional(),
});

// Logging configuration schema
export const LoggingConfigSchema = z.object({
  level: z.enum(['DEBUG', 'INFO', 'WARN', 'ERROR']).default('INFO'),
  format: z.enum(['json', 'text']).default('text'),
  file: z.string().optional(),
});

// Main server configuration schema
export const MCPServerConfigSchema = z.object({
  mcp: MCPConfigSchema,
  mlxEngine: MLXEngineConfigSchema,
  storage: StorageConfigSchema,
  analysis: AnalysisConfigSchema,
  security: SecurityConfigSchema,
  logging: LoggingConfigSchema,
});

/**
 * Validates and returns a typed configuration object
 * @param {unknown} config - Raw configuration object
 * @returns {object} Validated configuration
 */
export function validateConfig(config) {
  return MCPServerConfigSchema.parse(config);
}
