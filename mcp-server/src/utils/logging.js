/**
 * Logging utilities
 */

import * as log from '@std/log';

/**
 * Sets up logging configuration
 * @param {object} config - Logging configuration
 */
export function setupLogging(config) {
  const handlers = {
    console: new log.ConsoleHandler(config.level, {
      formatter: config.format === 'json' ? jsonFormatter : textFormatter,
    }),
  };

  // Add file handler if specified
  if (config.file) {
    handlers.file = new log.FileHandler(config.level, {
      filename: config.file,
      formatter: config.format === 'json' ? jsonFormatter : textFormatter,
    });
  }

  log.setup({
    handlers,
    loggers: {
      default: {
        level: config.level,
        handlers: Object.keys(handlers),
      },
    },
  });
}

/**
 * JSON log formatter
 * @param {object} logRecord - Log record
 * @returns {string} Formatted log message
 */
function jsonFormatter(logRecord) {
  return JSON.stringify({
    timestamp: logRecord.datetime.toISOString(),
    level: logRecord.levelName,
    message: logRecord.msg,
    logger: logRecord.loggerName,
  });
}

/**
 * Text log formatter
 * @param {object} logRecord - Log record
 * @returns {string} Formatted log message
 */
function textFormatter(logRecord) {
  const timestamp = logRecord.datetime.toISOString();
  return `${timestamp} [${logRecord.levelName}] ${logRecord.msg}`;
}

/**
 * Gets a logger instance
 * @param {string} name - Logger name
 * @returns {object} Logger instance
 */
export function getLogger(name = 'default') {
  return log.getLogger(name);
}
