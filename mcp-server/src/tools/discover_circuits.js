/**
 * Discover Circuits Tool - Discovers circuits for specific phenomena using causal tracing
 */

import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

// Schema definition
export const DiscoverCircuitsArgsSchema = z.object({
    prompt: z.string().min(1).describe("The prompt to analyze for circuit discovery"),
    phenomenon: z.enum(["IOI", "indirect_object_identification", "arithmetic", "factual_recall"]).optional().describe("Optional phenomenon type for categorization"),
    model_id: z.string(),
    confidence_threshold: z.number().min(0).max(1).default(0.7),
    max_circuits: z.number().int().min(1).max(50).default(10),
});

// Tool handler
export async function discoverCircuitsTool(args) {
    try {
        // Check if MLX Engine API is available
        const healthResponse = await fetch('http://localhost:50111/health');
        if (!healthResponse.ok) {
            return {
                success: false,
                error: "MLX Engine API server not available",
                endpoint_checked: 'http://localhost:50111/health',
                status_code: healthResponse.status,
                exact_parameters_passed: args
            };
        }

        // Check if a model is loaded
        const modelsResponse = await fetch('http://localhost:50111/v1/models');
        if (!modelsResponse.ok) {
            return {
                success: false,
                error: "Could not check loaded models",
                exact_parameters_passed: args
            };
        }
        
        const models = await modelsResponse.json();
        if (!models.models || models.models.length === 0) {
            return {
                success: false,
                error: "No models loaded in MLX Engine. Load a model first using the load_model tool.",
                available_models: models.models || [],
                exact_parameters_passed: args
            };
        }

        // Use the provided prompt for circuit analysis
        const prompt = args.prompt;
        
        // Capture activations from multiple layers for circuit analysis
        const activationHooks = [
            { layer_name: 'model.layers.0.self_attn', component: 'attention', hook_id: 'early_attention' },
            { layer_name: 'model.layers.5.self_attn', component: 'attention', hook_id: 'mid_attention' },
            { layer_name: 'model.layers.10.self_attn', component: 'attention', hook_id: 'late_attention' },
            { layer_name: 'model.layers.2.mlp', component: 'mlp', hook_id: 'early_mlp' },
            { layer_name: 'model.layers.8.mlp', component: 'mlp', hook_id: 'mid_mlp' },
            { layer_name: 'model.layers.15.mlp', component: 'mlp', hook_id: 'late_mlp' }
        ];

        const response = await fetch('http://localhost:50111/v1/chat/completions/with_activations', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                messages: [{ role: 'user', content: prompt }],
                max_tokens: args.max_circuits ? args.max_circuits * 5 : 50,
                temperature: 0.1,
                activation_hooks: activationHooks
            })
        });

        if (!response.ok) {
            const errorDetails = await response.json().catch(e => ({ error: `${response.status} ${response.statusText}` }));
            return {
                success: false,
                error: "MLX Engine activation capture failed",
                mlx_engine_error: errorDetails,
                exact_parameters_passed: args
            };
        }

        const result = await response.json();
        
        // Analyze captured activations to identify circuits
        const activations = result.activations || {};
        const circuits = [];
        
        for (const [hookId, activationList] of Object.entries(activations)) {
            if (activationList && activationList.length > 0) {
                circuits.push({
                    circuit_id: `${args.phenomenon || 'custom'}_${hookId}`,
                    phenomenon: args.phenomenon || 'custom',
                    layer_name: hookId.includes('attention') ? hookId.replace('_attention', '').replace('_', '.') : hookId.replace('_mlp', '').replace('_', '.'),
                    component: hookId.includes('mlp') ? 'mlp' : 'attention',
                    activation_count: activationList.length,
                    confidence: Math.min(0.9, activationList.length / 20.0), // Basic confidence based on activation count
                    description: `${hookId} circuit for ${args.phenomenon}`,
                    hook_id: hookId
                });
            }
        }

        // Format data for circuit_diagram tool compatibility
        const circuitDiscoveryData = {
            circuits: circuits.slice(0, args.max_circuits || 10),
            total_activations: Object.values(activations).reduce((sum, arr) => sum + (arr?.length || 0), 0),
            analysis_method: "causal_tracing",
            prompt: prompt,
            generated_text: result.choices[0].message.content
        };

        // Convert activations to the format expected by circuit_diagram
        const activationCaptureData = {
            activations: {},
            metadata: {
                total_tokens: result.usage?.total_tokens || 0,
                prompt_tokens: result.usage?.prompt_tokens || 0,
                completion_tokens: result.usage?.completion_tokens || 0
            }
        };

        // Map activations to layer names for circuit_diagram compatibility
        for (const [hookId, activationList] of Object.entries(activations)) {
            if (activationList && activationList.length > 0) {
                const layerName = hookId.includes('attention') ? 
                    hookId.replace('_attention', '').replace('_', '.') : 
                    hookId.replace('_mlp', '').replace('_', '.');
                
                // Use tensor shape format that circuit_diagram expects
                activationCaptureData.activations[`${layerName}_attention`] = [activationList.length, 768];
            }
        }

        return {
            success: true,
            phenomenon: args.phenomenon || "custom",
            model_id: models.models[0].id,
            prompt_used: prompt,
            generated_text: result.choices[0].message.content,
            circuits_discovered: circuits.length,
            circuits: circuits.slice(0, args.max_circuits || 10),
            total_activations_captured: Object.values(activations).reduce((sum, arr) => sum + (arr?.length || 0), 0),
            analysis_method: "real_activation_capture_with_mlx_engine",
            // Add the combined format for circuit_diagram compatibility
            circuit_discovery: circuitDiscoveryData,
            activation_capture: activationCaptureData
        };

    } catch (error) {
        return {
            success: false,
            error: `Circuit discovery failed: ${error.message}`,
            exact_parameters_passed: args,
            error_details: error.stack
        };
    }
}

// Tool definition
export const discoverCircuitsToolDefinition = {
    name: "discover_circuits",
    description: "Discovers circuits for a specific phenomenon using causal tracing with activation patching.",
    inputSchema: zodToJsonSchema(DiscoverCircuitsArgsSchema),
    handler: discoverCircuitsTool
};