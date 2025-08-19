/**
 * Configuration loader and manager
 */

import { readFile } from 'fs/promises';
import { validateConfig } from '../types/config.js';
import { defaultConfig } from './defaults.js';

/**
 * Loads API keys from the keys file
 * @returns {Promise<object>} API keys object
 */
async function loadApiKeys() {
  try {
    const keysContent = await readFile('/Users/craig/forks/Catalyst/keys', 'utf8');
    const keys = {};
    
    keysContent.split('\n').forEach(line => {
      const trimmed = line.trim();
      if (trimmed && !trimmed.startsWith('#')) {
        const [key, value] = trimmed.split('=');
        if (key && value) {
          keys[key.trim()] = value.trim();
        }
      }
    });
    

    return keys;
  } catch (error) {
    console.warn('Warning: Could not load keys file:', error.message);
    return {};
  }
}

/**
 * Loads configuration from environment variables and config files
 * @returns {Promise<object>} Validated configuration object
 */
export async function loadConfig() {
  // Start with default configuration
  let config = { ...defaultConfig };

  // Load API keys from keys file
  const apiKeys = await loadApiKeys();

  // Override with environment variables
  if (process.env.MCP_PORT) {
    config.mcp.port = parseInt(process.env.MCP_PORT, 10);
  }

  if (process.env.MCP_HOST) {
    config.mcp.host = process.env.MCP_HOST;
  }

  if (process.env.MLX_ENGINE_API_URL) {
    config.mlxEngine.apiUrl = process.env.MLX_ENGINE_API_URL;
  }

  if (process.env.MLX_ENGINE_API_KEY) {
    config.mlxEngine.apiKey = process.env.MLX_ENGINE_API_KEY;
  }

  if (process.env.LOG_LEVEL) {
    config.logging.level = process.env.LOG_LEVEL;
  }

  // Use API keys from keys file (environment variables take precedence)
  if (process.env.ANTHROPIC_API_KEY) {
    config.anthropic.apiKey = process.env.ANTHROPIC_API_KEY;
  } else if (apiKeys.anthropic) {
    config.anthropic.apiKey = apiKeys.anthropic;
  }

  if (!config.mlxEngine.apiKey && apiKeys.openai) {
    config.mlxEngine.apiKey = apiKeys.openai;
  }

  // Try to load from config file if it exists
  try {
    const configFile = await readFile('./config.json', 'utf8');
    const fileConfig = JSON.parse(configFile);
    config = { ...config, ...fileConfig };
  } catch (error) {
    // Config file is optional, continue with environment/defaults
    if (error.code !== 'ENOENT') {
      console.warn('Warning: Could not parse config.json:', error.message);
    }
  }

  // Validate and return configuration
  return validateConfig(config);
}
