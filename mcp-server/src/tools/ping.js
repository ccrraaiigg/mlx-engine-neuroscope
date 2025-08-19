/**
 * Ping Tool - Simple connectivity test tool
 */

import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

// Schema definition
export const PingArgsSchema = z.object({
    message: z.string().optional(),
});

// Tool handler
export async function pingTool(args) {
    const message = args.message || "Hello from Mechanistic Interpretability MCP Server!";
    return {
        success: true,
        message: `Ping received: ${message}`,
        timestamp: new Date().toISOString(),
        server: "mechanistic-interpretability-mcp-server",
        version: "0.1.0"
    };
}

// Tool definition
export const pingToolDefinition = {
    name: "ping",
    description: "Simple ping tool for testing server connectivity and basic functionality.",
    inputSchema: zodToJsonSchema(PingArgsSchema),
    handler: pingTool
};