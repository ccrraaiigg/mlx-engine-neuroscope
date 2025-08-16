/**
 * Logging utilities for Node.js
 */

// Simple logging implementation for Node.js
const loggers = new Map();

class Logger {
  constructor(name, level = 'INFO') {
    this.name = name;
    this.level = level;
    this.levels = {
      DEBUG: 0,
      INFO: 1,
      WARN: 2,
      ERROR: 3,
    };
  }

  shouldLog(level) {
    return this.levels[level] >= this.levels[this.level];
  }

  formatMessage(level, message) {
    const timestamp = new Date().toISOString();
    return `${timestamp} [${level}] [${this.name}] ${message}`;
  }

  debug(message) {
    if (this.shouldLog('DEBUG')) {
      console.debug(this.formatMessage('DEBUG', message));
    }
  }

  info(message) {
    if (this.shouldLog('INFO')) {
      console.info(this.formatMessage('INFO', message));
    }
  }

  warn(message) {
    if (this.shouldLog('WARN')) {
      console.warn(this.formatMessage('WARN', message));
    }
  }

  error(message) {
    if (this.shouldLog('ERROR')) {
      console.error(this.formatMessage('ERROR', message));
    }
  }
}

/**
 * Sets up logging configuration
 * @param {object} config - Logging configuration
 */
export function setupLogging(config) {
  // Store global logging config
  globalThis.LOGGING_CONFIG = config;
}

/**
 * Gets a logger instance
 * @param {string} name - Logger name
 * @returns {Logger} Logger instance
 */
export function getLogger(name = 'default') {
  if (!loggers.has(name)) {
    const config = globalThis.LOGGING_CONFIG || { level: 'INFO' };
    loggers.set(name, new Logger(name, config.level));
  }
  return loggers.get(name);
}
