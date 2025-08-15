/**
 * Default configuration values
 */

export const defaultConfig = {
  mcp: {
    port: 3000,
    host: 'localhost',
    maxConnections: 100,
  },

  mlxEngine: {
    apiUrl: 'http://localhost:8080',
    timeout: 30000,
    retryAttempts: 3,
    maxConcurrentRequests: 10,
  },

  storage: {
    activationsPath: './data/activations',
    circuitsPath: './data/circuits',
    cachePath: './data/cache',
    maxCacheSize: 1024 * 1024 * 1024, // 1GB
  },

  analysis: {
    defaultTimeout: 60000,
    cacheResults: true,
    maxCircuitCandidates: 100,
    confidenceThreshold: 0.8,
  },

  security: {
    enableAuth: false,
    rateLimiting: {
      windowMs: 15 * 60 * 1000, // 15 minutes
      maxRequests: 100,
    },
  },

  logging: {
    level: 'INFO',
    format: 'text',
  },
};
