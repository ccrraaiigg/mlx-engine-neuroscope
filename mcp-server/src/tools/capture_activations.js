import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema for capture_activations tool arguments
export const CaptureActivationsArgsSchema = z.object({
    prompt: z.string().min(1).max(10000),
    max_tokens: z.number().int().min(1).max(1000).default(100),
    temperature: z.number().min(0).max(2).default(0.7),
    capture_attention: z.boolean().default(true),
    capture_residual_stream: z.boolean().default(false),
});

// Handler function for capture_activations tool
export async function captureActivationsTool(args) {
    try {
        // Make real API call to MLX Engine
        const response = await fetch('http://localhost:50111/v1/chat/completions/with_activations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: args.prompt }],
                max_tokens: args.max_tokens,
                temperature: args.temperature,
                activation_hooks: [
                    { layer_name: 'model.layers.0.self_attn', component: 'attention' },
                    { layer_name: 'model.layers.5.self_attn', component: 'attention' },
                    { layer_name: 'model.layers.10.self_attn', component: 'attention' }
                ]
            })
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
                endpoint: '/v1/chat/completions/with_activations',
                request_parameters: {
                    prompt: args.prompt,
                    max_tokens: args.max_tokens,
                    temperature: args.temperature
                }
            };
            
            throw new Error(`MLX Engine API error: ${JSON.stringify(detailedError, null, 2)}`);
        }

        const result = await response.json();
        
        return {
            success: true,
            prompt: args.prompt,
            generated_text: result.choices[0].message.content,
            activations: result.activations,
            metadata: {
                total_tokens: result.usage.total_tokens,
                prompt_tokens: result.usage.prompt_tokens,
                completion_tokens: result.usage.completion_tokens,
                model_info: {
                    name: "gpt-oss-20b",
                    layers: 20,
                    hidden_size: 768
                }
            }
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            prompt: args.prompt
        };
    }
}

// Tool definition for MCP server
export const captureActivationsToolDefinition = {
    name: "capture_activations",
    description: "Captures activations during text generation for analysis.",
    inputSchema: zodToJsonSchema(CaptureActivationsArgsSchema),
};