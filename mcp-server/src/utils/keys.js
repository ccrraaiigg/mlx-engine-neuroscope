/**
 * API Keys utility for loading keys from the keys file
 */

import { getLogger } from './logging.js';

const logger = getLogger('Keys');

/**
 * Loads API keys from the keys file
 * @returns {Promise<Map<string, string>>} Map of service names to API keys
 */
export async function loadApiKeys() {
  const keys = new Map();
  
  try {
    const keysFile = await Deno.readTextFile('./keys');
    const lines = keysFile.split('\n').filter(line => line.trim() && !line.startsWith('#'));
    
    for (const line of lines) {
      const [service, key] = line.split('=', 2);
      if (service && key) {
        keys.set(service.trim(), key.trim());
        logger.info(`Loaded API key for service: ${service.trim()}`);
      }
    }
    
    logger.info(`Loaded ${keys.size} API keys`);
  } catch (error) {
    if (error instanceof Deno.errors.NotFound) {
      logger.warn('Keys file not found. API key functionality will be limited.');
    } else {
      logger.error(`Failed to load API keys: ${error.message}`);
    }
  }
  
  return keys;
}

/**
 * Gets a specific API key by service name
 * @param {string} service - Service name (e.g., 'anthropic', 'openai')
 * @returns {Promise<string|null>} API key or null if not found
 */
export async function getApiKey(service) {
  const keys = await loadApiKeys();
  return keys.get(service) || null;
}

/**
 * Checks if an API key exists for a service
 * @param {string} service - Service name
 * @returns {Promise<boolean>} True if key exists
 */
export async function hasApiKey(service) {
  const key = await getApiKey(service);
  return key !== null;
}

/**
 * Gets the Anthropic API key specifically
 * @returns {Promise<string|null>} Anthropic API key or null
 */
export async function getAnthropicApiKey() {
  return await getApiKey('anthropic');
}

/**
 * Gets the OpenAI API key specifically
 * @returns {Promise<string|null>} OpenAI API key or null
 */
export async function getOpenAiApiKey() {
  return await getApiKey('openai');
}

/**
 * Validates that required API keys are available
 * @param {string[]} requiredServices - Array of required service names
 * @throws {Error} If any required keys are missing
 */
export async function validateRequiredKeys(requiredServices) {
  const keys = await loadApiKeys();
  const missing = [];
  
  for (const service of requiredServices) {
    if (!keys.has(service)) {
      missing.push(service);
    }
  }
  
  if (missing.length > 0) {
    throw new Error(`Missing required API keys: ${missing.join(', ')}`);
  }
}