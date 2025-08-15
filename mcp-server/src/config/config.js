/**
 * Configuration loader and manager
 */

import { validateConfig } from '../types/config.js';
import { defaultConfig } from './defaults.js';

/**
 * Loads configuration from environment variables and config files
 * @returns {Promise<object>} Validated configuration object
 */
export async function loadConfig() {
  // Start with default configuration
  let config = { ...defaultConfig };

  // Override with environment variables
  if (Deno.env.get('MCP_PORT')) {
    config.mcp.port = parseInt(Deno.env.get('MCP_PORT'), 10);
  }

  if (Deno.env.get('MCP_HOST')) {
    config.mcp.host = Deno.env.get('MCP_HOST');
  }

  if (Deno.env.get('MLX_ENGINE_API_URL')) {
    config.mlxEngine.apiUrl = Deno.env.get('MLX_ENGINE_API_URL');
  }

  if (Deno.env.get('MLX_ENGINE_API_KEY')) {
    config.mlxEngine.apiKey = Deno.env.get('MLX_ENGINE_API_KEY');
  }

  if (Deno.env.get('LOG_LEVEL')) {
    config.logging.level = Deno.env.get('LOG_LEVEL');
  }

  // Try to load from config file if it exists
  try {
    const configFile = await Deno.readTextFile('./config.json');
    const fileConfig = JSON.parse(configFile);
    config = { ...config, ...fileConfig };
  } catch (error) {
    // Config file is optional, continue with environment/defaults
    if (!(error instanceof Deno.errors.NotFound)) {
      console.warn('Warning: Could not parse config.json:', error.message);
    }
  }

  // Validate and return configuration
  return validateConfig(config);
}
