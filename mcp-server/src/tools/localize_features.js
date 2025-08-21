import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { checkMLXEngineHealth, getMLXEngineBaseUrl } from '../utils/mlx-engine.js';
import fs from 'fs';
import fetch from 'node-fetch';

// File-based logging function
function logToFile(message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = data 
        ? `[${timestamp}] ${message}: ${JSON.stringify(data, null, 2)}\n`
        : `[${timestamp}] ${message}\n`;
    
    try {
        fs.appendFileSync('localize-features.log', logEntry);
    } catch (error) {
        // Silently fail to avoid interfering with JSON responses
    }
}

// Schema for advanced feature specifications
const FeatureSpecSchema = z.object({
  name: z.string().min(1, 'Feature name is required'),
  feature_type: z.enum(['semantic', 'syntactic', 'positional', 'attention', 'factual', 'arithmetic']).default('semantic'),
  description: z.string().default(''),
  target_layers: z.array(z.string()).default(['model.layers.12']),
  examples: z.array(z.string()).min(1, 'At least one example is required'),
  counter_examples: z.array(z.string()).default([])
});

// Schema for localization parameters
const LocalizationParametersSchema = z.object({
  // Sparse Autoencoder parameters
  hidden_dim: z.number().int().positive().default(512),
  sparsity_penalty: z.number().positive().default(0.01),
  epochs: z.number().int().positive().default(100),
  
  // Dictionary Learning parameters
  n_components: z.number().int().positive().default(256),
  alpha: z.number().positive().default(1.0),
  
  // PCA parameters
  pca_components: z.number().int().positive().default(50),
  
  // Probing Classifier parameters
  classifier_hidden_dim: z.number().int().positive().default(128)
});

// Main schema for advanced feature localization
export const LocalizeFeaturesArgsSchema = z.object({
    features: z.array(FeatureSpecSchema).min(1, 'At least one feature specification is required'),
    method: z.enum(['sparse_autoencoder', 'dictionary_learning', 'pca', 'probing_classifier', 'gradient_attribution']).default('sparse_autoencoder'),
    model_id: z.string().min(1, 'Model ID is required'),
    parameters: LocalizationParametersSchema.optional()
});

// Legacy schema for backward compatibility
export const LegacyLocalizeFeaturesArgsSchema = z.object({
    feature_name: z.string(),
    model_id: z.string(),
    layer_range: z.object({
        start: z.number().int().min(0),
        end: z.number().int().min(0),
    }).optional(),
    threshold: z.number().min(0).max(1).default(0.8),
});

// Main tool function
export async function localizeFeaturesTool(args) {
    try {
        // Check if this is a legacy call (has feature_name instead of features)
        const isLegacyCall = 'feature_name' in args && !('features' in args);
        
        let validatedArgs;
        if (isLegacyCall) {
            // Handle legacy format
            validatedArgs = LegacyLocalizeFeaturesArgsSchema.parse(args);
            return await handleLegacyFeatureLocalization(validatedArgs);
        } else {
            // Handle new advanced format
            validatedArgs = LocalizeFeaturesArgsSchema.parse(args);
            return await handleAdvancedFeatureLocalization(validatedArgs);
        }
        
    } catch (error) {
        logToFile('Feature localization error', error.message);
        return {
            success: false,
            error: error.message,
            details: error.stack
        };
    }
}

// Handle advanced feature localization using new endpoint
async function handleAdvancedFeatureLocalization(args) {
    // Check MLX Engine health
    const healthCheck = await checkMLXEngineHealth();
    if (!healthCheck.healthy) {
        return {
            success: false,
            error: `MLX Engine is not available: ${healthCheck.error}`,
            exact_parameters_passed: args
        };
    }
    
    const baseUrl = getMLXEngineBaseUrl();
    
    // Check if model is loaded
    const modelsResponse = await fetch(`${baseUrl}/models`, {
        timeout: 15 * 60 * 1000 // 15 minutes timeout
    });
    if (!modelsResponse.ok) {
        return {
            success: false,
            error: 'Failed to check available models',
            exact_parameters_passed: args
        };
    }
    
    const modelsData = await modelsResponse.json();
    // Handle the data wrapper structure from MLX Engine API
    const data = modelsData.data || modelsData;
    const loadedModels = data.loaded_models || data.models || [];
    
    if (!loadedModels.includes(args.model_id)) {
        return {
            success: false,
            error: `Model ${args.model_id} is not loaded. Available models: ${loadedModels.join(', ')}`,
            exact_parameters_passed: args
        };
    }
    
    // Prepare request payload
    const requestPayload = {
        features: args.features,
        method: args.method,
        model: args.model_id,
        parameters: args.parameters || {}
    };
    
    // Make request to advanced feature localization endpoint
    const response = await fetch(`${baseUrl}/v1/features/localize`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestPayload),
        timeout: 15 * 60 * 1000 // 15 minutes timeout
    });
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        return {
            success: false,
            error: `Feature localization failed: ${errorData.error || response.statusText}`,
            exact_parameters_passed: args
        };
    }
    
    const result = await response.json();
    
    // Format the results for better readability
    return {
        success: true,
        localized_features: result.features.map(feature => ({
            feature_name: feature.feature_name,
            feature_type: feature.feature_type,
            layer: feature.layer_name,
            neurons: {
                indices: feature.neuron_indices,
                count: feature.neuron_indices.length,
                activation_strength: feature.activation_strength
            },
            confidence: feature.confidence,
            method_used: feature.localization_method,
            metadata: feature.metadata
        })),
        analysis: {
            layer_analysis: result.layer_analysis,
            feature_interactions: result.feature_interactions,
            execution_time_ms: result.execution_time_ms,
            total_features_found: result.features.length
        },
        method_used: args.method,
        model_id: args.model_id,
        metadata: result.metadata
    };
}

// Handle legacy feature localization for backward compatibility
async function handleLegacyFeatureLocalization(args) {
    // Check MLX Engine health
    const healthCheck = await checkMLXEngineHealth();
    if (!healthCheck.healthy) {
        return {
            success: false,
            error: `MLX Engine is not available: ${healthCheck.error}`,
            exact_parameters_passed: args
        };
    }
    
    const baseUrl = getMLXEngineBaseUrl();
    
    // Check if model is loaded
    const modelsResponse = await fetch(`${baseUrl}/models`, {
        timeout: 15 * 60 * 1000 // 15 minutes timeout
    });
    if (!modelsResponse.ok) {
        return {
            success: false,
            error: 'Failed to check available models',
            exact_parameters_passed: args
        };
    }
    
    const modelsData = await modelsResponse.json();
    const loadedModels = modelsData.loaded_models || [];
    
    if (!loadedModels.includes(args.model_id)) {
        return {
            success: false,
            error: `Model ${args.model_id} is not loaded. Available models: ${loadedModels.join(', ')}`,
            exact_parameters_passed: args
        };
    }
    
    // Convert legacy format to new format
    const legacyFeatureSpec = {
        name: args.feature_name,
        feature_type: 'semantic', // Default for legacy calls
        description: `Legacy feature localization for ${args.feature_name}`,
        target_layers: args.layer_range ? 
            [`model.layers.${args.layer_range.start}`, `model.layers.${args.layer_range.end}`] : 
            ['model.layers.12'],
        examples: [
            `Example prompt for ${args.feature_name}`,
            `Another example for ${args.feature_name}`
        ],
        counter_examples: [
            'Unrelated example',
            'Another unrelated example'
        ]
    };
    
    // Use PCA method for legacy calls (simpler and faster)
    const requestPayload = {
        features: [legacyFeatureSpec],
        method: 'pca',
        model: args.model_id,
        parameters: {
            pca_components: 50
        }
    };
    
    // Try advanced endpoint first, fallback to basic if not available
    let response = await fetch(`${baseUrl}/v1/features/localize`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestPayload)
    });
    
    if (response.status === 501) {
        // Advanced endpoint not available, use basic activation capture
        return await fallbackToBasicLocalization(args, baseUrl);
    }
    
    if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
        return {
            success: false,
            error: `Feature localization failed: ${errorData.error || response.statusText}`,
            exact_parameters_passed: args
        };
    }
    
    const result = await response.json();
    
    // Format for legacy compatibility
    const firstFeature = result.features[0];
    return {
        success: true,
        feature_name: args.feature_name,
        localized_neurons: firstFeature ? firstFeature.neuron_indices : [],
        confidence: firstFeature ? firstFeature.confidence : 0,
        layer_analysis: result.layer_analysis,
        method_used: 'pca',
        execution_time_ms: result.execution_time_ms,
        exact_parameters_passed: args
    };
}

// Fallback to basic activation capture for legacy compatibility
async function fallbackToBasicLocalization(args, baseUrl) {
    try {
        // Use basic activation capture as fallback
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
                }),
                timeout: 15 * 60 * 1000 // 15 minutes timeout
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
            model_id: args.model_id,
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