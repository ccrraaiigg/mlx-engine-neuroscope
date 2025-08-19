#!/usr/bin/env node

/**
 * MCP Server Restart Validation Script
 * 
 * This script helps AI assistants determine if an MCP server restart
 * is required before testing functionality.
 */

const fs = require('fs');
const path = require('path');

// Files that require MCP restart when modified
const RESTART_REQUIRED_FILES = [
    'src/tools/*.js',
    'src/services/*.js', 
    'src/mcp-server.js',
    'server.js',
    'package.json',
    'package-lock.json'
];

// Check if any restart-required files were recently modified
function checkRecentModifications(minutesThreshold = 30) {
    const now = Date.now();
    const threshold = minutesThreshold * 60 * 1000;
    
    const recentlyModified = [];
    
    // Check each pattern
    RESTART_REQUIRED_FILES.forEach(pattern => {
        if (pattern.includes('*')) {
            // Handle glob patterns
            const dir = path.dirname(pattern);
            const ext = path.extname(pattern);
            
            if (fs.existsSync(dir)) {
                const files = fs.readdirSync(dir)
                    .filter(file => file.endsWith(ext))
                    .map(file => path.join(dir, file));
                    
                files.forEach(file => {
                    const stats = fs.statSync(file);
                    if (now - stats.mtime.getTime() < threshold) {
                        recentlyModified.push({
                            file,
                            modified: stats.mtime,
                            minutesAgo: Math.round((now - stats.mtime.getTime()) / 60000)
                        });
                    }
                });
            }
        } else {
            // Handle direct file paths
            if (fs.existsSync(pattern)) {
                const stats = fs.statSync(pattern);
                if (now - stats.mtime.getTime() < threshold) {
                    recentlyModified.push({
                        file: pattern,
                        modified: stats.mtime,
                        minutesAgo: Math.round((now - stats.mtime.getTime()) / 60000)
                    });
                }
            }
        }
    });
    
    return recentlyModified;
}

// Generate restart reminder message
function generateRestartReminder(modifiedFiles) {
    if (modifiedFiles.length === 0) {
        return {
            restartNeeded: false,
            message: "✅ No recent MCP server file modifications detected. Safe to test."
        };
    }
    
    const fileList = modifiedFiles
        .map(f => `  - ${f.file} (${f.minutesAgo} minutes ago)`)
        .join('\n');
        
    return {
        restartNeeded: true,
        message: `🔄 MCP SERVER RESTART REQUIRED\n\nRecently modified files:\n${fileList}\n\n⚠️ STOP: Do not test MCP functionality until restart confirmed!\n\nUser must restart MCP server in IDE before proceeding.`
    };
}

// Main validation function
function validateRestartStatus() {
    console.log('🔍 Checking MCP server restart requirements...');
    
    const modifiedFiles = checkRecentModifications(30);
    const result = generateRestartReminder(modifiedFiles);
    
    console.log(result.message);
    
    return result;
}

// Export for use in other scripts
module.exports = {
    checkRecentModifications,
    generateRestartReminder,
    validateRestartStatus,
    RESTART_REQUIRED_FILES
};

// Run if called directly
if (require.main === module) {
    const result = validateRestartStatus();
    process.exit(result.restartNeeded ? 1 : 0);
}