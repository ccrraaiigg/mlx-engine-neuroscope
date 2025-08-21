import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import fetch from 'node-fetch';

// Schema for available_models tool arguments
export const AvailableModelsArgsSchema = z.object({
    include_loaded: z.boolean().default(true),
    include_available: z.boolean().default(true),
});

// Handler function for available_models tool
export async function availableModelsTool(args) {
    try {
        // Make real API call to MLX Engine
        const response = await fetch('http://localhost:50111/v1/models/available', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            },
            timeout: 15 * 60 * 1000 // 15 minutes timeout
        });

        if (!response.ok) {
            let errorDetails;
            try {
                errorDetails = await response.json();
            } catch (e) {
                errorDetails = { error: `${response.status} ${response.statusText}` };
            }
            
            // Create detailed error message with MLX Engine's enhanced error info
            const detailedError = {
                mlx_engine_error: errorDetails,
                status_code: response.status,
                status_text: response.statusText,
                endpoint: '/v1/models/available',
                request_parameters: {
                    include_loaded: args.include_loaded,
                    include_available: args.include_available
                }
            };
            
            throw new Error(`MLX Engine API error: ${JSON.stringify(detailedError, null, 2)}`);
        }

        const result = await response.json();
        
        // Handle response data structure (MLX Engine wraps data in 'data' object)
        const data = result.data || result;
        
        return {
            success: true,
            available_models: args.include_available ? data.available_models || [] : [],
            loaded_models: args.include_loaded ? data.loaded_models || [] : [],
            current_model: data.current_model || null,
            search_paths: data.search_paths || [],
            total_available: data.total_available || 0,
            total_loaded: data.total_loaded || 0,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            available_models: [],
            loaded_models: [],
            current_model: null,
            search_paths: [],
            total_available: 0,
            total_loaded: 0,
            timestamp: new Date().toISOString()
        };
    }
}

// Tool definition for MCP server
export const availableModelsToolDefinition = {
    name: "available_models",
    description: "Gets available and loaded models from the MLX Engine.",
    inputSchema: zodToJsonSchema(AvailableModelsArgsSchema),
};