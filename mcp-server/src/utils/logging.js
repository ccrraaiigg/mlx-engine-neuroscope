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
      // Always log to stderr to avoid interfering with stdio JSON-RPC streams
      process.stderr.write(this.formatMessage('DEBUG', message) + '\n');
    }
  }

  info(message) {
    if (this.shouldLog('INFO')) {
      process.stderr.write(this.formatMessage('INFO', message) + '\n');
    }
  }

  warn(message) {
    if (this.shouldLog('WARN')) {
      process.stderr.write(this.formatMessage('WARN', message) + '\n');
    }
  }

  error(message) {
    if (this.shouldLog('ERROR')) {
      process.stderr.write(this.formatMessage('ERROR', message) + '\n');
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
