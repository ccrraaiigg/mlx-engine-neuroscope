/**
 * Version Tool - Returns server version information
 */

import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Helper function to get version from package.json
export function getServerVersion() {
    const packageJsonPath = path.join(__dirname, '../../package.json');
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    return packageJson.version;
}

// Schema definition
export const VersionArgsSchema = z.object({
    random_string: z.string().optional(),
});

// Tool handler
export async function versionTool(args) {
    return {
        success: true,
        version: getServerVersion(),
        server: "mechanistic-interpretability-mcp-server",
        timestamp: new Date().toISOString(),
        capabilities: [
            "circuit_discovery",
            "feature_localization", 
            "activation_capture",
            "visualization",
            "neuroscope_export"
        ]
    };
}

// Tool definition
export const versionToolDefinition = {
    name: "version",
    description: "Returns the current version of the MCP server to track updates.",
    inputSchema: zodToJsonSchema(VersionArgsSchema),
    handler: versionTool
};
