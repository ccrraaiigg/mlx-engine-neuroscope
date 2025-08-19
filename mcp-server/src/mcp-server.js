#!/usr/bin/env node

/**
 * Mechanistic Interpretability MCP Server
 * Modeled exactly after the working filesystem server pattern
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, ToolSchema } from "@modelcontextprotocol/sdk/types.js";
import { initializeMLXClient } from './services/mlx_tools.js';
import { writeFileReadOnly, writeFile, makeFileReadOnly } from './utils/file_utils.js';
import { TOOL_DEFINITIONS, TOOL_HANDLERS, getToolHandler, validateToolArgs } from './tools/index.js';
import { getServerVersion } from './tools/version.js';
import { healthCheckTool } from './tools/health_check.js';
import { startServerTool } from './tools/start_server.js';
import { loadConfig } from './config/config.js';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

// Command line argument parsing (optional for this server)
const args = process.argv.slice(2);

// Initialize the MCP server
const server = new Server({
    name: "mechanistic-interpretability-mcp-server",
    version: getServerVersion()
}, {
    capabilities: {
        tools: {},
    },
});

// Use only the modular tools from the tools directory
const TOOLS = TOOL_DEFINITIONS;

// List tools handler
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: TOOLS,
    };
});

// Call tool handler - simplified to only use modular tools
server.setRequestHandler(CallToolRequestSchema, async (request) => {
    const { name, arguments: args } = request.params;
    
    try {
        // Try to get modular tool handler
        const handler = getToolHandler(name);
        if (handler) {
            // Validate arguments using modular schema
            const validatedArgs = validateToolArgs(name, args || {});
            
            // For describe_tools, we need to pass the TOOLS array
            let result;
            if (name === 'describe_tools') {
                result = await handler(validatedArgs, TOOLS);
            } else {
                result = await handler(validatedArgs);
            }
            
            // Return result in proper MCP format with content array
            return {
                content: [
                    {
                        type: "text",
                        text: typeof result === 'string' ? result : JSON.stringify(result, null, 2)
                    }
                ]
            };
        }
        
        // If no handler found, return error
        throw new Error(`Unknown tool: ${name}`);
        
    } catch (error) {
        return {
            content: [
                {
                    type: "text", 
                    text: JSON.stringify({
                        success: false,
                        error: error.message,
                        details: error.stack
                    }, null, 2)
                }
            ],
            isError: true
        };
    }
});

// Function to start required services
async function startRequiredServices() {
    try {
        console.error('🚀 Starting required services...');
        
        // Start MLX Engine service
        console.error('   🔧 Starting MLX Engine service...');
        const mlxResult = await startServerTool({ service: 'mlx', force: true });
        if (!mlxResult.success) {
            throw new Error(`Failed to start MLX Engine: ${mlxResult.error}`);
        }
        console.error('   ✅ MLX Engine service started successfully');
        
        // Start visualization server
        console.error('   🎨 Starting visualization server...');
        const vizResult = await startServerTool({ service: 'visualization', force: true });
        if (!vizResult.success) {
            throw new Error(`Failed to start visualization server: ${vizResult.error}`);
        }
        console.error('   ✅ Visualization server started successfully');
        
        // Wait a moment for services to fully initialize
        await new Promise(resolve => setTimeout(resolve, 3000));
        console.error('🚀 All services started successfully');
        
    } catch (error) {
        console.error('❌ Error starting services:', error.message);
        throw error;
    }
}

// Cleanup function for existing servers
async function cleanupExistingServers() {
    const fs = await import('fs');
    const path = await import('path');
    const { exec } = await import('child_process');
    const { promisify } = await import('util');
    const execAsync = promisify(exec);
    
    try {
        console.error('🧹 Cleaning up existing services...');
        
        // Kill MLX Engine service processes
        try {
            await execAsync('pkill -f "mlx_engine_service.py"');
            console.error('   ✅ Killed MLX Engine service processes');
        } catch (e) {
            console.error('   ℹ️  No MLX Engine service processes found');
        }
        
        // Kill MLX Engine API REST server processes
        try {
            await execAsync('pkill -f "api_server.py"');
            console.error('   ✅ Killed MLX Engine API REST server processes');
        } catch (e) {
            console.error('   ℹ️  No MLX Engine API REST server processes found');
        }
        
        // Kill visualization server processes
        try {
            await execAsync('pkill -f "visualization.*server"');
            console.error('   ✅ Killed visualization server processes');
        } catch (e) {
            console.error('   ℹ️  No visualization server processes found');
        }
        
        // Kill any processes using the specific ports
        try {
            await execAsync('lsof -ti:50111 | xargs kill -9');
            console.error('   ✅ Freed port 50111 (MLX Engine)');
        } catch (e) {
            console.error('   ℹ️  Port 50111 already free');
        }
        
        try {
            await execAsync('lsof -ti:8888 | xargs kill -9');
            console.error('   ✅ Freed port 8888 (Visualization)');
        } catch (e) {
            console.error('   ℹ️  Port 8888 already free');
        }
        
        // Clean up PID files
        try {
            const pidFile = path.join(process.cwd(), 'mlx_engine_service.pid');
            if (fs.existsSync(pidFile)) {
                fs.unlinkSync(pidFile);
                console.error('   ✅ Removed MLX Engine PID file');
            }
        } catch (e) {
            console.error('   ⚠️  Could not remove PID file:', e.message);
        }
        
        // Clean up old visualization files
        try {
            const visualizationDir = path.join(process.cwd(), 'mcp-server', 'src', 'visualization');
            const filesToCleanup = [
                path.join(visualizationDir, 'real_circuit.html'),
                path.join(visualizationDir, 'real_circuit_data.json')
            ];
            
            for (const filePath of filesToCleanup) {
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                    console.error(`   ✅ Removed old visualization file: ${path.basename(filePath)}`);
                }
            }
        } catch (e) {
            console.error('   ⚠️  Could not remove visualization files:', e.message);
        }
        
        // Wait for cleanup to complete
        await new Promise(resolve => setTimeout(resolve, 2000));
        console.error('🧹 Cleanup completed');
        
    } catch (error) {
        console.error('❌ Error during cleanup:', error.message);
        throw error;
    }
}

async function performStartupHealthCheck() {
    try {
        console.error(`🏥 Performing startup health check...`);
        
        // Run health check for all services
        const healthResult = await healthCheckTool({ service: 'all' });
        
        if (healthResult.success && healthResult.overall_status === 'healthy') {
            console.error(`✅ Health check completed: ${healthResult.overall_status}`);
            
            // Log individual service statuses
            if (healthResult.services.mlx_engine) {
                const mlxStatus = healthResult.services.mlx_engine.status;
                console.error(`   🔧 MLX Engine: ${mlxStatus}`);
            }
            
            if (healthResult.services.visualization_server) {
                const vizStatus = healthResult.services.visualization_server.status;
                console.error(`   📊 Visualization Server: ${vizStatus}`);
            }
            
            return true;
        } else {
            console.error(`❌ Health check failed: ${healthResult.error || 'Services not healthy'}`);
            
            // Log individual service statuses
            if (healthResult.services && healthResult.services.mlx_engine) {
                const mlxStatus = healthResult.services.mlx_engine.status;
                console.error(`   🔧 MLX Engine: ${mlxStatus}`);
            }
            
            if (healthResult.services && healthResult.services.visualization_server) {
                const vizStatus = healthResult.services.visualization_server.status;
                console.error(`   📊 Visualization Server: ${vizStatus}`);
            }
            
            // Fail startup if health check doesn't pass
            throw new Error(`Health check failed: ${healthResult.overall_status || 'Services not healthy'}`);
        }
    } catch (error) {
        console.error(`❌ Health check error: ${error.message}`);
        throw error; // Re-throw to fail startup
    }
}

// Cleanup function for visualization files
async function cleanupVisualizationFiles() {
    try {
        const __filename = fileURLToPath(import.meta.url);
        const __dirname = path.dirname(__filename);
        const visualizationDir = path.join(__dirname, 'visualization');
        const filesToDelete = [
            path.join(visualizationDir, 'real_circuit_data.json'),
            path.join(visualizationDir, 'real_circuit.html')
        ];
        
        for (const filePath of filesToDelete) {
            try {
                if (fs.existsSync(filePath)) {
                    fs.unlinkSync(filePath);
                    console.error(`🧹 Deleted old visualization file: ${path.basename(filePath)}`);
                }
            } catch (error) {
                console.error(`⚠️  Could not delete ${path.basename(filePath)}: ${error.message}`);
            }
        }
    } catch (error) {
        console.error(`⚠️  Error during visualization cleanup: ${error.message}`);
    }
}

async function runServer() {
    try {
        // Report server version on startup
        const version = getServerVersion();
        console.error(`🚀 Starting Mechanistic Interpretability MCP Server v${version}`);
        console.error(`📊 Server: mechanistic-interpretability-mcp-server`);
        console.error(`⏰ Startup time: ${new Date().toISOString()}`);
        
        // Load configuration
        const config = await loadConfig();
        console.error(`⚙️  Configuration loaded successfully`);
        
        // Clean up visualization files from previous runs
        await cleanupVisualizationFiles();
        
        // Clean up any existing servers
        await cleanupExistingServers();
        
        // Start required services
        await startRequiredServices();
        
        // Initialize MLX client with configuration
        await initializeMLXClient(config.mlxEngine);
        console.error(`🔧 MLX client initialized`);
        
        // Perform health check on startup - this will fail startup if services aren't healthy
        await performStartupHealthCheck();
        
        // Connect MCP server transport
        const transport = new StdioServerTransport();
        await server.connect(transport);
        
        console.error(`✅ Mechanistic Interpretability MCP Server v${version} running on stdio`);
        console.error(`🎯 All services verified healthy and ready`);
        
        // Handle graceful shutdown
        process.on('SIGINT', async () => {
            console.error('\n🛑 Received SIGINT, shutting down server...');
            await cleanupExistingServers();
            process.exit(0);
        });
        
        process.on('SIGTERM', async () => {
            console.error('\n🛑 Received SIGTERM, shutting down server...');
            await cleanupExistingServers();
            process.exit(0);
        });
        
    } catch (error) {
        console.error("❌ Failed to start MCP server:", error.message);
        console.error("🧹 Cleaning up services before exit...");
        
        // Attempt cleanup before exiting
        try {
            await cleanupExistingServers();
        } catch (cleanupError) {
            console.error("⚠️  Cleanup failed:", cleanupError.message);
        }
        
        process.exit(1);
    }
}

runServer().catch((error) => {
    console.error("Fatal error:", error);
    process.exit(1);
});

