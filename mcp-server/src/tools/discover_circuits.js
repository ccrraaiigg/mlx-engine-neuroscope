/**
 * Discover Circuits Tool - Discovers circuits for specific phenomena using causal tracing
 */

import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';
import fs from 'fs';
import path from 'path';
import fetch from 'node-fetch';

// File-based logging function to avoid interfering with JSON responses
function logToFile(message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = data 
        ? `${timestamp} [DISCOVER_CIRCUITS] ${message}: ${JSON.stringify(data, null, 2)}\n`
        : `${timestamp} [DISCOVER_CIRCUITS] ${message}\n`;
    
    fs.appendFileSync('/Users/craig/me/behavior/forks/mlx-engine-neuroscope/mcp-server/discover-circuits.log', logEntry);
}

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
    logToFile('=== DISCOVER CIRCUITS TOOL START ===');
        logToFile('Input arguments', args);
        logToFile('Timestamp', new Date().toISOString());
    
    try {
        // Check if MLX Engine API is available
        logToFile('Checking MLX Engine API health at http://localhost:50111/health');
        const healthResponse = await fetch('http://localhost:50111/health', {
            timeout: 30 * 60 * 1000 // 30 minutes timeout
        });
        logToFile('Health check response status', healthResponse.status);
        logToFile('Health check response headers', Object.fromEntries(healthResponse.headers.entries()));
        
        if (!healthResponse.ok) {
            const healthText = await healthResponse.text();
            logToFile('Health check failed. Response body', healthText);
            return {
                success: false,
                error: "MLX Engine API server not available",
                endpoint_checked: 'http://localhost:50111/health',
                status_code: healthResponse.status,
                response_body: healthText,
                exact_parameters_passed: args
            };
        }
        
        const healthData = await healthResponse.json();
        logToFile('Health check successful. Response data', healthData);

        // Check if a model is loaded
        logToFile('Checking loaded models at http://localhost:50111/v1/models');
        const modelsResponse = await fetch('http://localhost:50111/v1/models', {
            timeout: 30 * 60 * 1000 // 30 minutes timeout
        });
        logToFile('Models check response status', modelsResponse.status);
        logToFile('Models check response headers', Object.fromEntries(modelsResponse.headers.entries()));
        
        if (!modelsResponse.ok) {
            const modelsText = await modelsResponse.text();
            logToFile('Models check failed. Response body', modelsText);
            return {
                success: false,
                error: "Could not check loaded models",
                response_body: modelsText,
                exact_parameters_passed: args
            };
        }
        
        const models = await modelsResponse.json();
        logToFile('Models check successful. Response data', models);
        
        // Handle the data wrapper structure from MLX Engine API
        const modelsList = models.data?.models || models.models || [];
        
        if (!modelsList || modelsList.length === 0) {
            logToFile('No models loaded in MLX Engine');
            return {
                success: false,
                error: "No models loaded in MLX Engine. Load a model first using the load_model tool.",
                available_models: modelsList,
                exact_parameters_passed: args
            };
        }
        
        logToFile('Found loaded models', modelsList.map(m => m.model_id || m.id || m.name || m));

        // Use the provided prompt for circuit analysis
        const prompt = args.prompt;
        logToFile('Using prompt for analysis', prompt);
        
        // Prepare request payload for enhanced circuit discovery
        const requestPayload = {
            prompt: prompt,
            phenomenon: args.phenomenon || 'general',
            confidence_threshold: args.confidence_threshold || 0.7,
            max_circuits: args.max_circuits || 10,
            model: args.model_id,
            // Enhanced causal tracing parameters
            noise_type: 'gaussian',
            noise_scale: 0.1,
            noise_samples: 10,
            attribution_method: 'integrated_gradients',
            attribution_steps: 50,
            baseline_strategy: 'zero',
            significance_threshold: 0.05,
            multiple_comparisons: 'bonferroni',
            bootstrap_samples: 1000
        };
        
        logToFile('=== ENHANCED CIRCUIT DISCOVERY REQUEST ===');
        logToFile('Endpoint: http://localhost:50111/v1/circuits/enhanced-discover');
        logToFile('Method: POST');
        logToFile('Request payload', requestPayload);
        logToFile('Request timestamp', new Date().toISOString());
        
        // Try enhanced circuit discovery first (with noise injection and statistical analysis)
        let response;
        try {
            logToFile('Initiating fetch request...');
            
            response = await fetch('http://localhost:50111/v1/circuits/enhanced-discover', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(requestPayload),
                timeout: 30 * 60 * 1000 // 30 minutes timeout
            });
            
            logToFile('Fetch request completed successfully');
            logToFile('Response status', response.status);
            logToFile('Response status text', response.statusText);
            logToFile('Response headers', Object.fromEntries(response.headers.entries()));
        } catch (fetchError) {
            logToFile('=== FETCH ERROR OCCURRED ===');
            logToFile('Error type', fetchError.constructor.name);
            logToFile('Error message', fetchError.message);
            logToFile('Error stack', fetchError.stack);
            logToFile('Error details', JSON.stringify(fetchError, Object.getOwnPropertyNames(fetchError), 2));
            throw fetchError;
        }
        
        // If enhanced discovery is not available, try regular circuit discovery
        if (!response.ok && response.status === 501) {
            logToFile('Enhanced circuit discovery not available, trying regular circuit discovery');
            
            response = await fetch('http://localhost:50111/v1/circuits/discover', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    prompt: prompt,
                    phenomenon: args.phenomenon || 'general',
                    confidence_threshold: args.confidence_threshold || 0.7,
                    max_circuits: args.max_circuits || 10,
                    model: args.model_id
                }),
                timeout: 30 * 60 * 1000 // 30 minutes timeout
            });
        }
        
        // If advanced discovery is not available, fall back to basic activation capture
        if (!response.ok && response.status === 501) {
            logToFile('Advanced circuit discovery not available, falling back to basic activation capture');
            
            // Capture activations from multiple layers for circuit analysis
            const activationHooks = [
                { layer_name: 'model.layers.0.self_attn', component: 'attention', hook_id: 'early_attention' },
                { layer_name: 'model.layers.5.self_attn', component: 'attention', hook_id: 'mid_attention' },
                { layer_name: 'model.layers.10.self_attn', component: 'attention', hook_id: 'late_attention' },
                { layer_name: 'model.layers.2.mlp', component: 'mlp', hook_id: 'early_mlp' },
                { layer_name: 'model.layers.8.mlp', component: 'mlp', hook_id: 'mid_mlp' },
                { layer_name: 'model.layers.15.mlp', component: 'mlp', hook_id: 'late_mlp' }
            ];

            response = await fetch('http://localhost:50111/v1/chat/completions/with_activations', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    messages: [{ role: 'user', content: prompt }],
                    max_tokens: args.max_circuits ? args.max_circuits * 5 : 50,
                    temperature: 0.1,
                    activation_hooks: activationHooks
                }),
                timeout: 30 * 60 * 1000 // 30 minutes timeout
            });
        }

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
        
        // Check if this is an enhanced circuit discovery response
        if (result.success && result.enhanced_circuits) {
            // Enhanced circuit discovery response with statistical analysis
            return {
                success: true,
                phenomenon: args.phenomenon || result.phenomenon || "custom",
                model_id: args.model_id,
                prompt_used: prompt,
                circuits_discovered: result.enhanced_circuits.length,
                circuits: result.enhanced_circuits.slice(0, args.max_circuits || 10),
                analysis_method: "enhanced_causal_tracing_with_noise_injection",
                statistical_analysis: result.statistical_analysis,
                noise_robustness: result.noise_robustness,
                attribution_scores: result.attribution_scores,
                mediation_analysis: result.mediation_analysis,
                confidence_intervals: result.confidence_intervals,
                circuit_discovery: {
                    circuits: result.enhanced_circuits,
                    analysis_method: "enhanced_causal_tracing",
                    prompt: prompt,
                    phenomenon: result.phenomenon,
                    statistical_significance: result.statistical_analysis,
                    noise_robustness: result.noise_robustness
                }
            };
        }
        
        // Check if this is a regular advanced circuit discovery response
        if (result.success && result.circuits) {
            // Regular advanced circuit discovery response
            return {
                success: true,
                phenomenon: args.phenomenon || result.phenomenon || "custom",
                model_id: args.model_id,
                prompt_used: prompt,
                circuits_discovered: result.circuit_count || result.circuits.length,
                circuits: result.circuits.slice(0, args.max_circuits || 10),
                analysis_method: "advanced_activation_patching_with_causal_tracing",
                circuit_discovery: {
                    circuits: result.circuits,
                    analysis_method: "activation_patching",
                    prompt: prompt,
                    phenomenon: result.phenomenon
                }
            };
        }
        
        // Fallback: Basic activation capture response
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
            analysis_method: "basic_activation_capture",
            prompt: prompt,
            generated_text: result.choices?.[0]?.message?.content || ''
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
            model_id: modelsList[0].model_id || modelsList[0].id,
            prompt_used: prompt,
            generated_text: result.choices?.[0]?.message?.content || '',
            circuits_discovered: circuits.length,
            circuits: circuits.slice(0, args.max_circuits || 10),
            total_activations_captured: Object.values(activations).reduce((sum, arr) => sum + (arr?.length || 0), 0),
            analysis_method: "basic_activation_capture_with_mlx_engine",
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
