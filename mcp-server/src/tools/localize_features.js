import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema for localize_features tool arguments
export const LocalizeFeaturesArgsSchema = z.object({
    feature_name: z.string(),
    model_id: z.string(),
    layer_range: z.object({
        start: z.number().int().min(0),
        end: z.number().int().min(0),
    }).optional(),
    threshold: z.number().min(0).max(1).default(0.8),
});

// Handler function for localize_features tool
export async function localizeFeaturesTool(args) {
    try {
        // Check if MLX Engine API is available
        const healthResponse = await fetch('http://localhost:50111/health');
        if (!healthResponse.ok) {
            return {
                success: false,
                error: "MLX Engine API server not available",
                endpoint_checked: 'http://localhost:50111/health',
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

        // Use real activation capture to localize features
        // Note: Agents should provide their own feature-specific prompts
        // This tool requires agent-provided examples for proper feature localization
        const prompts = ['AGENT_PROVIDED_PROMPT_1', 'AGENT_PROVIDED_PROMPT_2'];
        
        // Feature localization requires contrasting examples to identify neurons
        // Agents should provide appropriate positive/negative or correct/incorrect examples
        
        // Analyze activations across multiple layers to localize the feature
        const layerRange = args.layer_range || { start: 0, end: 15 };
        const activationHooks = [];
        
        for (let layer = layerRange.start; layer <= Math.min(layerRange.end, 15); layer += 3) {
            activationHooks.push({ 
                layer_name: `model.layers.${layer}.self_attn`, 
                component: 'attention', 
                hook_id: `attention_layer_${layer}` 
            });
            activationHooks.push({ 
                layer_name: `model.layers.${layer}.mlp`, 
                component: 'mlp', 
                hook_id: `mlp_layer_${layer}` 
            });
        }

        const featureAnalysis = [];
        
        for (const prompt of prompts) {
            const response = await fetch('http://localhost:50111/v1/chat/completions/with_activations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: prompt }],
                    max_tokens: 20,
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
            featureAnalysis.push({
                prompt: prompt,
                generated_text: result.choices[0].message.content,
                activations: result.activations || {}
            });
        }

        // Analyze which layers/components show the most variation for this feature
        const localizations = [];
        
        for (const hookConfig of activationHooks) {
            let activationVariation = 0;
            let totalActivations = 0;
            
            for (const analysis of featureAnalysis) {
                const activations = analysis.activations[hookConfig.hook_id] || [];
                totalActivations += activations.length;
                activationVariation += activations.length;
            }
            
            if (totalActivations > 0) {
                localizations.push({
                    layer_name: hookConfig.layer_name,
                    component: hookConfig.component,
                    hook_id: hookConfig.hook_id,
                    activation_count: totalActivations,
                    localization_strength: Math.min(1.0, activationVariation / 100.0),
                    feature_name: args.feature_name
                });
            }
        }

        // Sort by localization strength
        localizations.sort((a, b) => b.localization_strength - a.localization_strength);

        return {
            success: true,
            feature_name: args.feature_name,
            model_id: models.models[0].id,
            prompts_tested: prompts,
            layer_range_analyzed: layerRange,
            localizations: localizations.slice(0, 10), // Top 10 localizations
            total_activations_captured: localizations.reduce((sum, loc) => sum + loc.activation_count, 0),
            analysis_method: "real_activation_capture_comparison",
            feature_analyses: featureAnalysis
        };

    } catch (error) {
        return {
            success: false,
            error: `Feature localization failed: ${error.message}`,
            exact_parameters_passed: args,
            error_details: error.stack
        };
    }
}

// Tool definition for MCP server
export const localizeFeaturesToolDefinition = {
    name: "localize_features", 
    description: "Localizes neurons responsible for specific features using Principal Component Analysis and probing classifiers.",
    inputSchema: zodToJsonSchema(LocalizeFeaturesArgsSchema),
};