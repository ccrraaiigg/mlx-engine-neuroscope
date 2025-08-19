import fs from 'fs';
import { getLogger } from './logging.js';

const logger = getLogger('file_utils');

/**
 * Makes a file read-only by setting appropriate permissions
 * @param {string} filePath - Path to the file to make read-only
 * @returns {Promise<boolean>} - True if successful, false otherwise
 */
export async function makeFileReadOnly(filePath) {
  try {
    // Set file permissions to read-only (444 = r--r--r--)
    await fs.promises.chmod(filePath, 0o444);
    logger.info(`File made read-only: ${filePath}`);
    return true;
  } catch (error) {
    logger.error(`Failed to make file read-only: ${filePath}`, error);
    return false;
  }
}

/**
 * Writes content to a file normally (without read-only protection)
 * @param {string} filePath - Path to the file to write
 * @param {string|Buffer} content - Content to write to the file
 * @param {object} options - Options for fs.writeFile
 * @returns {Promise<boolean>} - True if successful, false otherwise
 */
export async function writeFile(filePath, content, options = {}) {
  try {
    // If file exists and is read-only, make it writable first
    try {
      await fs.promises.access(filePath, fs.constants.F_OK);
      // File exists, make it writable
      await fs.promises.chmod(filePath, 0o644);
      logger.info(`Made existing file writable: ${filePath}`);
    } catch (accessError) {
      // File doesn't exist, that's fine
    }
    
    // Write the file (but don't make it read-only)
    await fs.promises.writeFile(filePath, content, options);
    logger.info(`File written: ${filePath}`);
    
    return true;
  } catch (error) {
    logger.error(`Failed to write file: ${filePath}`, error);
    return false;
  }
}

/**
 * Writes content to a file and then makes it read-only
 * @param {string} filePath - Path to the file to write
 * @param {string|Buffer} content - Content to write to the file
 * @param {object} options - Options for fs.writeFile
 * @returns {Promise<boolean>} - True if successful, false otherwise
 */
export async function writeFileReadOnly(filePath, content, options = {}) {
  try {
    // If file exists and is read-only, make it writable first
    try {
      await fs.promises.access(filePath, fs.constants.F_OK);
      // File exists, make it writable
      await fs.promises.chmod(filePath, 0o644);
      logger.info(`Made existing file writable: ${filePath}`);
    } catch (accessError) {
      // File doesn't exist, that's fine
    }
    
    // Write the file
    await fs.promises.writeFile(filePath, content, options);
    logger.info(`File written: ${filePath}`);
    
    // Then make it read-only
    const readOnlySuccess = await makeFileReadOnly(filePath);
    
    return readOnlySuccess;
  } catch (error) {
    logger.error(`Failed to write file: ${filePath}`, error);
    return false;
  }
}

/**
 * Writes content to a file synchronously and then makes it read-only
 * @param {string} filePath - Path to the file to write
 * @param {string|Buffer} content - Content to write to the file
 * @param {object} options - Options for fs.writeFileSync
 * @returns {boolean} - True if successful, false otherwise
 */
export function writeFileReadOnlySync(filePath, content, options = {}) {
  try {
    // Write the file first
    fs.writeFileSync(filePath, content, options);
    logger.info(`File written (sync): ${filePath}`);
    
    // Then make it read-only
    fs.chmodSync(filePath, 0o444);
    logger.info(`File made read-only (sync): ${filePath}`);
    
    return true;
  } catch (error) {
    logger.error(`Failed to write file (sync): ${filePath}`, error);
    return false;
  }
}