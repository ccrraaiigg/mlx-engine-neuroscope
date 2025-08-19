import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema definition
export const StartServerArgsSchema = z.object({
    service: z.enum(["mlx", "visualization"]),
    force: z.boolean().default(false),
});

// Tool definition
export const startServerToolDefinition = {
    name: "start_server",
    description: "Start MLX Engine or Visualization server if not running.",
    inputSchema: zodToJsonSchema(StartServerArgsSchema),
};

// Tool implementation
export async function startServerTool(args) {
    const { spawn } = await import('child_process');
    const path = await import('path');
    
    try {
        if (args.service === 'mlx') {
            // Check if already running (unless force is true)
            if (!args.force) {
                try {
                    const response = await fetch('http://localhost:50111/health');
                    if (response.ok) {
                        return {
                            success: false,
                            error: "MLX Engine is already running. Use force=true to restart.",
                            already_running: true
                        };
                    }
                } catch (e) {
                    // Server not running, proceed to start
                }
            }
            
            // Start MLX Engine service  
            const projectRoot = '/Users/craig/me/behavior/forks/mlx-engine-neuroscope';
            const scriptPath = path.join(projectRoot, 'mcp-server', 'mlx_engine_service.py');
            console.error("Current working directory:", process.cwd());
            console.error("Attempting to start MLX Engine with script:", scriptPath);
            const child = spawn('python3', [scriptPath], {
                detached: true,
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: projectRoot
            });
            
            child.unref();
            
            // Wait a moment for startup
            await new Promise(resolve => setTimeout(resolve, 3000));
            
            // Verify it started
            try {
                const response = await fetch('http://localhost:50111/health');
                if (response.ok) {
                    return {
                        success: true,
                        service: 'MLX Engine',
                        pid: child.pid,
                        url: 'http://localhost:50111',
                        message: 'MLX Engine started successfully'
                    };
                }
            } catch (e) {
                return {
                    success: false,
                    error: 'MLX Engine failed to start or is not responding',
                    pid: child.pid
                };
            }
            
        } else if (args.service === 'visualization') {
            // Check if already running (unless force is true)
            if (!args.force) {
                try {
                    const response = await fetch('http://localhost:8888/health');
                    if (response.ok) {
                        return {
                            success: false,
                            error: "Visualization server is already running. Use force=true to restart.",
                            already_running: true
                        };
                    }
                } catch (e) {
                    // Server not running, proceed to start
                }
            }
            
            // Start visualization server
            const projectRoot = '/Users/craig/me/behavior/forks/mlx-engine-neuroscope';
            const scriptPath = path.join(projectRoot, 'mcp-server', 'src', 'visualization', 'server.js');
            console.error("Current working directory:", process.cwd());
            console.error("Attempting to start visualization server with script:", scriptPath);
            const child = spawn('/opt/homebrew/bin/node', [scriptPath], {
                detached: true,
                stdio: ['ignore', 'pipe', 'pipe'],
                cwd: path.join(projectRoot, 'mcp-server', 'src', 'visualization')
            });
            
            child.unref();
            
            // Wait a moment for startup
            await new Promise(resolve => setTimeout(resolve, 2000));
            
            // Verify it started
            try {
                const response = await fetch('http://localhost:8888/health');
                if (response.ok) {
                    return {
                        success: true,
                        service: 'Visualization Server',
                        pid: child.pid,
                        url: 'http://localhost:8888',
                        message: 'Visualization server started successfully'
                    };
                }
            } catch (e) {
                return {
                    success: false,
                    error: 'Visualization server failed to start or is not responding',
                    pid: child.pid
                };
            }
        }
        
        return {
            success: false,
            error: `Unknown service: ${args.service}`
        };
        
    } catch (error) {
        return {
            success: false,
            error: error.message,
            service: args.service
        };
    }
}