/**
 * MLX Engine Utility Functions
 * 
 * Provides utility functions for interacting with the MLX Engine API,
 * including health checks and configuration management.
 */

import { getLogger } from './logging.js';

const logger = getLogger('MLXEngineUtils');

// Default MLX Engine configuration
const DEFAULT_MLX_CONFIG = {
  apiUrl: 'http://localhost:8080',
  timeout: 30000,
  retryAttempts: 3
};

/**
 * Gets the base URL for the MLX Engine API
 * @param {object} config - Optional configuration override
 * @returns {string} Base URL for MLX Engine API
 */
export function getMLXEngineBaseUrl(config = {}) {
  const baseUrl = config.apiUrl || process.env.MLX_ENGINE_URL || DEFAULT_MLX_CONFIG.apiUrl;
  logger.debug(`Using MLX Engine base URL: ${baseUrl}`);
  return baseUrl;
}

/**
 * Checks the health status of the MLX Engine
 * @param {object} config - Optional configuration override
 * @returns {Promise<object>} Health status response
 */
export async function checkMLXEngineHealth(config = {}) {
  const baseUrl = getMLXEngineBaseUrl(config);
  const timeout = config.timeout || DEFAULT_MLX_CONFIG.timeout;
  
  try {
    logger.debug('Checking MLX Engine health...');
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(`${baseUrl}/health`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }
    
    const healthData = await response.json();
    logger.info('MLX Engine health check successful');
    
    return {
      success: true,
      status: 'healthy',
      data: healthData,
      baseUrl: baseUrl
    };
    
  } catch (error) {
    logger.error(`MLX Engine health check failed: ${error.message}`);
    
    return {
      success: false,
      status: 'unhealthy',
      error: error.message,
      baseUrl: baseUrl
    };
  }
}

/**
 * Makes a request to the MLX Engine API
 * @param {string} endpoint - API endpoint path
 * @param {object} options - Request options
 * @param {object} config - MLX Engine configuration
 * @returns {Promise<object>} Response data
 */
export async function makeMLXEngineRequest(endpoint, options = {}, config = {}) {
  const baseUrl = getMLXEngineBaseUrl(config);
  const timeout = config.timeout || DEFAULT_MLX_CONFIG.timeout;
  const url = `${baseUrl}${endpoint}`;
  
  const requestOptions = {
    method: options.method || 'GET',
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  };

  // Add request body if provided
  if (options.body && typeof options.body === 'object') {
    requestOptions.body = JSON.stringify(options.body);
  }

  try {
    logger.debug(`Making request to ${url}`);
    
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), timeout);
    
    const response = await fetch(url, {
      ...requestOptions,
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    const data = await response.json();
    logger.debug(`Request successful: ${endpoint}`);
    return data;
    
  } catch (error) {
    logger.error(`Request failed: ${endpoint} - ${error.message}`);
    throw error;
  }
}

/**
 * Validates MLX Engine connection and configuration
 * @param {object} config - MLX Engine configuration
 * @returns {Promise<object>} Validation result
 */
export async function validateMLXEngineConnection(config = {}) {
  try {
    const healthCheck = await checkMLXEngineHealth(config);
    
    if (!healthCheck.success) {
      return {
        valid: false,
        error: `MLX Engine is not accessible: ${healthCheck.error}`,
        config: config
      };
    }
    
    return {
      valid: true,
      status: healthCheck.status,
      data: healthCheck.data,
      config: config
    };
    
  } catch (error) {
    return {
      valid: false,
      error: `Connection validation failed: ${error.message}`,
      config: config
    };
  }
}

/**
 * Gets MLX Engine configuration from environment or defaults
 * @returns {object} MLX Engine configuration
 */
export function getMLXEngineConfig() {
  return {
    apiUrl: process.env.MLX_ENGINE_URL || DEFAULT_MLX_CONFIG.apiUrl,
    timeout: parseInt(process.env.MLX_ENGINE_TIMEOUT) || DEFAULT_MLX_CONFIG.timeout,
    retryAttempts: parseInt(process.env.MLX_ENGINE_RETRY_ATTEMPTS) || DEFAULT_MLX_CONFIG.retryAttempts,
    apiKey: process.env.MLX_ENGINE_API_KEY || null
  };
}