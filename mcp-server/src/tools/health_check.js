import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema definition
export const HealthCheckArgsSchema = z.object({
    service: z.enum(["mlx", "visualization", "all"]).default("all"),
});

// Tool definition
export const healthCheckToolDefinition = {
    name: "health_check",
    description: "Check health status of MLX Engine and Visualization servers.",
    inputSchema: zodToJsonSchema(HealthCheckArgsSchema),
};

// Tool implementation
export async function healthCheckTool(args) {
    const results = {};
    
    async function checkMLXEngine() {
        try {
            const response = await fetch('http://localhost:50111/health');
            if (response.ok) {
                const data = await response.json();
                return {
                    status: 'healthy',
                    reachable: true,
                    response_time: Date.now(),
                    ...data
                };
            } else {
                return {
                    status: 'unhealthy',
                    reachable: true,
                    http_status: response.status,
                    error: `HTTP ${response.status}`
                };
            }
        } catch (error) {
            return {
                status: 'unreachable',
                reachable: false,
                error: error.message
            };
        }
    }
    
    async function checkVisualizationServer() {
        try {
            const response = await fetch('http://localhost:8888/health');
            if (response.ok) {
                const data = await response.json();
                return {
                    status: 'healthy',
                    reachable: true,
                    response_time: Date.now(),
                    ...data
                };
            } else {
                return {
                    status: 'unhealthy',
                    reachable: true,
                    http_status: response.status,
                    error: `HTTP ${response.status}`
                };
            }
        } catch (error) {
            return {
                status: 'unreachable',
                reachable: false,
                error: error.message
            };
        }
    }
    
    if (args.service === 'mlx' || args.service === 'all') {
        results.mlx_engine = await checkMLXEngine();
    }
    
    if (args.service === 'visualization' || args.service === 'all') {
        results.visualization_server = await checkVisualizationServer();
    }
    
    const allHealthy = Object.values(results).every(r => r.status === 'healthy');
    
    return {
        success: true,
        overall_status: allHealthy ? 'healthy' : 'degraded',
        timestamp: new Date().toISOString(),
        services: results
    };
}