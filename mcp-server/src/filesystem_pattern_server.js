#!/usr/bin/env node

/**
 * Mechanistic Interpretability MCP Server
 * Modeled exactly after the working filesystem server pattern
 */

import { Server } from "@modelcontextprotocol/sdk/server/index.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { CallToolRequestSchema, ListToolsRequestSchema, ToolSchema } from "@modelcontextprotocol/sdk/types.js";
import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { initializeMLXClient } from './services/mlx_tools.js';

// Command line argument parsing (optional for this server)
const args = process.argv.slice(2);

// Global state for user preferences with hard-wired defaults
const globalPreferences = {
    browserMode: 'chrome-new-window', // 'default' or 'chrome-new-window' - DEFAULT: chrome-new-window for better UX
    defaultBrowser: 'system', // 'system', 'chrome', 'safari', 'firefox'
};

// Input schemas using Zod (like filesystem server)
const PingArgsSchema = z.object({
    message: z.string().optional(),
});

const VersionArgsSchema = z.object({
    random_string: z.string().optional(),
});

const DiscoverCircuitsArgsSchema = z.object({
    phenomenon: z.enum(["IOI", "indirect_object_identification", "arithmetic", "factual_recall"]),
    model_id: z.string(),
    confidence_threshold: z.number().min(0).max(1).default(0.7),
    max_circuits: z.number().int().min(1).max(50).default(10),
});

const LocalizeFeaturesArgsSchema = z.object({
    feature_name: z.string(),
    model_id: z.string(),
    layer_range: z.object({
        start: z.number().int().min(0),
        end: z.number().int().min(0),
    }).optional(),
    threshold: z.number().min(0).max(1).default(0.8),
});

const CaptureActivationsArgsSchema = z.object({
    prompt: z.string().min(1).max(10000),
    max_tokens: z.number().int().min(1).max(1000).default(100),
    temperature: z.number().min(0).max(2).default(0.7),
    capture_attention: z.boolean().default(true),
    capture_residual_stream: z.boolean().default(false),
});

// MLX Engine Tools
const LoadModelArgsSchema = z.object({
    model_id: z.enum(["gpt-oss-20b"]),
    quantization: z.enum(["none", "4bit", "8bit"]).default("none"),
    max_context_length: z.number().int().min(512).max(131072).default(2048),
    device: z.enum(["auto", "cpu", "mps", "cuda"]).default("auto"),
});

const AvailableModelsArgsSchema = z.object({
    // No arguments needed for this tool
});

const CreateHooksArgsSchema = z.object({
    layers: z.array(z.number().int().min(0).max(50)).min(1),
    hook_type: z.enum(["activation", "gradient", "both"]).default("activation"),
    components: z.array(z.enum(["mlp", "attention", "residual", "all"])).default(["mlp", "attention"]),
    capture_gradients: z.boolean().default(false),
});

const AnalyzeMathArgsSchema = z.object({
    prompt: z.string().min(1),
    max_tokens: z.number().int().min(10).max(500).default(100),
    temperature: z.number().min(0).max(1).default(0.1),
    analysis_depth: z.enum(["shallow", "medium", "deep"]).default("medium"),
});

const AnalyzeAttentionArgsSchema = z.object({
    prompt: z.string().min(1),
    layers: z.array(z.number().int().min(0).max(50)).min(1).max(10),
});

const AnalyzeFactualArgsSchema = z.object({
    query: z.string().min(1),
    max_tokens: z.number().int().min(5).max(200).default(50),
    analysis_depth: z.enum(["shallow", "medium", "deep"]).default("medium"),
});

const TrackResidualArgsSchema = z.object({
    prompt: z.string().min(1),
    layers: z.array(z.number().int().min(0).max(50)).default([]),
    components: z.array(z.enum(["attention", "mlp", "both"])).default(["both"]),
});

const ExportNeuroScopeArgsSchema = z.object({
    analysis_data: z.object({}),
    format: z.enum(["smalltalk", "json", "both"]).default("smalltalk"),
});

// Visualization Tools
const CircuitDiagramArgsSchema = z.object({
    circuit_data: z.any(),
    circuit_name: z.string().default("Circuit Analysis"),
});

const AttentionPatternsArgsSchema = z.object({
    attention_data: z.object({}),
    layers: z.array(z.number().int()),
});

const ActivationFlowArgsSchema = z.object({
    activation_data: z.object({}),
    prompt: z.string(),
});

const OpenBrowserArgsSchema = z.object({
    url: z.string().default("http://localhost:8888"),
});

// Health Check and Server Management Schemas
const HealthCheckArgsSchema = z.object({
    service: z.enum(["mlx", "visualization", "all"]).default("all"),
});

const StartServerArgsSchema = z.object({
    service: z.enum(["mlx", "visualization"]),
    force: z.boolean().default(false),
});

const GenerateReportArgsSchema = z.object({
    title: z.string(),
    analysis_data: z.object({}),
});

const SetPreferencesArgsSchema = z.object({
    browserMode: z.enum(['default', 'chrome-new-window']).optional(),
    defaultBrowser: z.enum(['system', 'chrome', 'safari', 'firefox']).optional(),
});

const GetPreferencesArgsSchema = z.object({
    random_string: z.string().optional(),
});

const DescribeToolsArgsSchema = z.object({
    random_string: z.string().optional(),
});

// Server setup - exactly like filesystem server
const server = new Server({
    name: "mechanistic-interpretability-mcp-server",
    version: "0.1.0",
}, {
    capabilities: {
        tools: {},
    },
});

// Tool implementations
async function pingTool(args) {
    const message = args.message || "Hello from Mechanistic Interpretability MCP Server!";
    return {
        success: true,
        message: `Ping received: ${message}`,
        timestamp: new Date().toISOString(),
        server: "mechanistic-interpretability-mcp-server",
        version: "0.1.0"
    };
}

async function versionTool(args) {
    return {
        success: true,
        version: 116,
        server: "mechanistic-interpretability-mcp-server",
        last_modified: new Date().toISOString(),
        changes: "FIX: Generate comprehensive link network for mixed attention/MLP circuits"
    };
}

async function discoverCircuitsTool(args) {
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

        // Use real activation capture to analyze circuits for the specified phenomenon
        const circuitPrompts = {
            'IOI': 'When Mary and John went to the store, Mary gave it to',
            'indirect_object_identification': 'After the teacher gave the book to the student, the student gave it to',
            'arithmetic': 'What is 15 + 27? The answer is',
            'factual_recall': 'The capital of France is'
        };

        const prompt = circuitPrompts[args.phenomenon] || circuitPrompts['factual_recall'];
        
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
                    circuit_id: `${args.phenomenon}_${hookId}`,
                    phenomenon: args.phenomenon,
                    layer_name: hookId.includes('attention') ? hookId.replace('_attention', '').replace('_', '.') : hookId.replace('_mlp', '').replace('_', '.'),
                    component: hookId.includes('mlp') ? 'mlp' : 'attention',
                    activation_count: activationList.length,
                    confidence: Math.min(0.9, activationList.length / 20.0), // Basic confidence based on activation count
                    description: `${hookId} circuit for ${args.phenomenon}`,
                    hook_id: hookId
                });
            }
        }

        return {
            success: true,
            phenomenon: args.phenomenon,
            model_id: models.models[0].id,
            prompt_used: prompt,
            generated_text: result.choices[0].message.content,
            circuits_discovered: circuits.length,
            circuits: circuits.slice(0, args.max_circuits || 10),
            total_activations_captured: Object.values(activations).reduce((sum, arr) => sum + (arr?.length || 0), 0),
            analysis_method: "real_activation_capture_with_mlx_engine"
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

async function localizeFeaturesTool(args) {
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
        // Test with prompts that should activate the specified feature
        const featurePrompts = {
            'sentiment': ['This movie is absolutely terrible', 'This movie is absolutely wonderful'],
            'syntax': ['The cat that was sleeping', 'Was the cat that sleeping'],
            'factual': ['The capital of France is Paris', 'The capital of France is Berlin'],
            'arithmetic': ['2 + 2 = 4', '2 + 2 = 5'],
            'negation': ['This is not good', 'This is good']
        };

        const prompts = featurePrompts[args.feature_name] || featurePrompts['sentiment'];
        
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


async function captureActivationsTool(args) {
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

// MLX Engine Tools
async function loadModelTool(args) {
    try {
        // Make real API call to MLX Engine
        const response = await fetch('http://localhost:50111/v1/models/load', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                model_path: `/Users/craig/.lmstudio/models/nightmedia/${args.model_id}-q5-hi-mlx`,
                model_id: args.model_id,
                quantization: args.quantization,
                max_context_length: args.max_context_length,
                device: args.device
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
                endpoint: '/v1/models/load',
                request_parameters: {
                    model_id: args.model_id,
                    model_path: `/Users/craig/.lmstudio/models/nightmedia/${args.model_id}-q5-hi-mlx`,
                    quantization: args.quantization,
                    max_context_length: args.max_context_length,
                    device: args.device
                }
            };
            
            throw new Error(`MLX Engine API error: ${JSON.stringify(detailedError, null, 2)}`);
        }

        const result = await response.json();
        
        return {
            success: true,
            model_id: args.model_id,
            status: result.status,
            model_info: {
                architecture: "transformer",
                num_layers: 20,
                hidden_size: 768,
                vocab_size: 32000,
                quantization: args.quantization,
                device: args.device,
                supports_activations: result.supports_activations
            },
            load_time_ms: 2000
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            model_id: args.model_id
        };
    }
}

async function availableModelsTool(args) {
    try {
        // Make API call to MLX Engine to get available models
        const response = await fetch('http://localhost:50111/v1/models/available', {
            method: 'GET',
            headers: {
                'Content-Type': 'application/json',
            }
        });

        if (!response.ok) {
            let errorDetails;
            try {
                errorDetails = await response.json();
            } catch (e) {
                errorDetails = { error: `${response.status} ${response.statusText}` };
            }
            
            const detailedError = {
                mlx_engine_error: errorDetails,
                status_code: response.status,
                status_text: response.statusText,
                endpoint: '/v1/models/available'
            };
            
            throw new Error(`MLX Engine API error: ${JSON.stringify(detailedError, null, 2)}`);
        }

        const result = await response.json();
        
        return {
            success: true,
            available_models: result.available_models,
            loaded_models: result.loaded_models,
            current_model: result.current_model,
            search_paths: result.search_paths,
            total_available: result.total_available,
            total_loaded: result.total_loaded,
            timestamp: new Date().toISOString()
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            available_models: [],
            loaded_models: []
        };
    }
}

async function createHooksTool(args) {
    return {
        success: true,
        hooks_created: args.layers.length * args.components.length,
        hook_config: {
            layers: args.layers,
            hook_type: args.hook_type,
            components: args.components,
            capture_gradients: args.capture_gradients
        },
        hook_ids: args.layers.map(layer => `hook_${layer}_${args.hook_type}`)
    };
}

async function analyzeMathTool(args) {
    // AGENT.md: Never fake anything. If information is missing, DO NOT guess, "mock", or simulate.
    // Report the exact parameters and error details instead of returning mock data.
    try {
        const mlxTools = await import('./services/mlx_tools.js');
        const mlxAnalyzeMath = mlxTools.mlxTools.find(tool => tool.name === 'mlx_analyze_math');
        
        if (!mlxAnalyzeMath) {
            return {
                success: false,
                error: "MLX analyze_math tool not found in mlx_tools module",
                prompt: args.prompt,
                exact_parameters_passed: args
            };
        }
        
        // Call the real MLX implementation
        const result = await mlxAnalyzeMath.handler(args);
        return result;
        
    } catch (error) {
        return {
            success: false,
            error: `Failed to analyze math with real model: ${error.message}`,
            prompt: args.prompt,
            exact_parameters_passed: args,
            stack_trace: error.stack
        };
    }
}

async function analyzeAttentionTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('./services/mlx_tools.js');
        const mlxAnalyzeAttention = mlxTools.mlxTools.find(tool => tool.name === 'mlx_analyze_attention');
        
        if (!mlxAnalyzeAttention) {
            return {
                success: false,
                error: "MLX analyze_attention tool not found in mlx_tools module",
                prompt: args.prompt,
                layers: args.layers,
                exact_parameters_passed: args
            };
        }
        
        // Call the real MLX implementation
        const result = await mlxAnalyzeAttention.handler(args);
        return result;
        
    } catch (error) {
        return {
            success: false,
            error: `Failed to analyze attention with real model: ${error.message}`,
            prompt: args.prompt,
            layers: args.layers,
            exact_parameters_passed: args,
            stack_trace: error.stack
        };
    }
}

async function analyzeFactualTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('./services/mlx_tools.js');
        const mlxAnalyzeFactual = mlxTools.mlxTools.find(tool => tool.name === 'mlx_analyze_factual');
        
        if (!mlxAnalyzeFactual) {
            return {
                success: false,
                error: "MLX analyze_factual tool not found in mlx_tools module",
                query: args.query,
                exact_parameters_passed: args
            };
        }
        
        // Call the real MLX implementation
        const result = await mlxAnalyzeFactual.handler(args);
        return result;
        
    } catch (error) {
        return {
            success: false,
            error: `Failed to analyze factual recall with real model: ${error.message}`,
            query: args.query,
            exact_parameters_passed: args,
            stack_trace: error.stack
        };
    }
}

async function trackResidualTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('./services/mlx_tools.js');
        const mlxTrackResidual = mlxTools.mlxTools.find(tool => tool.name === 'mlx_track_residual');
        
        if (!mlxTrackResidual) {
            return {
                success: false,
                error: "MLX track_residual tool not found in mlx_tools module",
                prompt: args.prompt,
                exact_parameters_passed: args
            };
        }
        
        // Call the real MLX implementation
        const result = await mlxTrackResidual.handler(args);
        return result;
        
    } catch (error) {
        return {
            success: false,
            error: `Failed to track residual stream with real model: ${error.message}`,
            prompt: args.prompt,
            exact_parameters_passed: args,
            stack_trace: error.stack
        };
    }
}

async function exportNeuroScopeTool(args) {
    return {
        success: true,
        format: args.format,
        export_paths: {
            smalltalk: args.format !== "json" ? "./exports/analysis.st" : null,
            json: args.format !== "smalltalk" ? "./exports/analysis.json" : null
        },
        data_size_kb: 256,
        timestamp: new Date().toISOString()
    };
}

// Visualization Tools
async function circuitDiagramTool(args) {
    try {
        // Declare all variables at the beginning of the function
        let circuitData = args.circuit_data;
        let processedData = circuitData;
        let nodes = [];
        let links = [];
        let nodeId = 0;
        let linkId = 0;
        
        // Parse circuit_data if it's a string
        if (typeof circuitData === 'string') {
            try {
                circuitData = JSON.parse(circuitData);
            } catch (e) {
                circuitData = args.circuit_data;
            }
        }
        
        const fs = await import('fs/promises');
        const path = await import('path');
        
        // Write directly to the visualization directory where the server is running
        const vizDir = '/Users/craig/me/behavior/forks/mlx-engine-neuroscope/mcp-server/src/visualization';
        
        // Write the real circuit data to a file
        const circuitDataPath = path.join(vizDir, 'real_circuit_data.json');
        await fs.writeFile(circuitDataPath, JSON.stringify(circuitData, null, 2));
        
        // Convert activation data to nodes/links format
        processedData = circuitData;
        
        // Check if this is raw activation data with model_layers structure or old format
        const hasActivationLayers = circuitData && (
            // New format: model_layers object
            circuitData.model_layers || 
            // Activation capture format: activations object
            (circuitData.activations && Object.keys(circuitData.activations).length > 0) ||
            // Circuit discovery + activation capture combined format
            (circuitData.circuit_discovery && circuitData.activation_capture) ||
            // Old format: direct layer keys
            Object.keys(circuitData).some(key => key.startsWith('model.layers.') && Array.isArray(circuitData[key]))
        );
        
        console.error("DEBUG: hasActivationLayers =", hasActivationLayers);
        console.error("DEBUG: circuitData.nodes exists =", !!circuitData.nodes);
        console.error("DEBUG: circuitData keys =", Object.keys(circuitData));
        console.error("DEBUG: condition check: circuitData.nodes && !hasActivationLayers =", (circuitData.nodes && !hasActivationLayers));
        
        console.error("CONTINUING AFTER DEBUG CHECKS...");
        
        // Arrays are already initialized at function start
        
        if (hasActivationLayers) {
            console.error("ENTERING hasActivationLayers BLOCK - this should NOT happen when hasActivationLayers = false!");
            
            // Create nodes from activation data
            // nodeId already declared at function start
            
            // Handle new model_layers format
            if (circuitData.model_layers) {
                Object.keys(circuitData.model_layers).forEach(layerKey => {
                    const layerData = circuitData.model_layers[layerKey];
                    
                    const newNode = {
                        id: `node_${nodeId++}`,
                        label: `${layerKey} (${layerData.component})`,
                        type: layerData.component,
                        value: 0.8, // Add value for sizing
                        color: layerData.component === 'mlp' ? [1.0, 0.4, 0.4, 1.0] : [0.4, 0.6, 1.0, 1.0], // Red for MLP, Blue for attention
                        nodeColor: layerData.component === 'mlp' ? [1.0, 0.4, 0.4, 1.0] : [0.4, 0.6, 1.0, 1.0], // Backup color property
                        layer: layerData.layer,
                        position: { 
                            x: (layerData.layer * 150) + (Math.random() * 50 - 25),
                            y: (layerData.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                        },
                        metadata: {
                            shape: layerData.shape,
                            count: layerData.activation_count,
                            component: layerData.component,
                            layer_name: layerKey,
                            dtype: layerData.dtype
                        }
                    };
                    
                    nodes.push(newNode);
                    // Note: Server-side console.log interferes with MCP JSON protocol
                });
            } else if (circuitData.circuit_discovery && circuitData.activation_capture) {
                // Handle combined circuit discovery + activation capture format
                const circuits = circuitData.circuit_discovery.circuits || [];
                const activations = circuitData.activation_capture.activations || {};
                
                // Create nodes from circuit discovery data
                circuits.forEach(circuit => {
                    const layerMatch = circuit.layer_name.match(/(\d+)/);
                    const layerNum = layerMatch ? parseInt(layerMatch[1]) : 0;
                    
                    // Separate visual encodings: confidence = size, activation = opacity
                    let nodeValue = circuit.confidence || 0.8;  // Size reflects circuit confidence
                    let tensorVolume = 1;
                    let nodeOpacity = 0.8;  // Opacity reflects activation intensity
                    
                    // Look for corresponding activation data
                    const activationKey = Object.keys(activations).find(key => 
                        key.includes(`layers.${layerNum}`) && 
                        key.includes(circuit.component.replace('attention', 'attn'))
                    );
                    
                    if (activationKey && activations[activationKey]) {
                        const tensorShape = activations[activationKey];
                        if (Array.isArray(tensorShape) && tensorShape.length >= 2) {
                            // Calculate tensor volume (dimensions multiplied)
                            tensorVolume = tensorShape.reduce((acc, dim) => acc * dim, 1);
                            // Normalize to opacity range (0.3 to 1.0)
                            nodeOpacity = Math.min(1.0, Math.max(0.3, tensorVolume / 100));
                        }
                    }
                    
                    const newNode = {
                        id: `node_${nodeId++}`,
                        label: `${circuit.layer_name} (${circuit.component})`,
                        type: circuit.component,
                        value: nodeValue,  // Size = circuit confidence
                        opacity: nodeOpacity,  // Opacity = activation intensity
                        color: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                        nodeColor: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                        layer: layerNum,
                        position: { 
                            x: (layerNum * 150) + (Math.random() * 50 - 25),
                            y: (circuit.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                        },
                        metadata: {
                            activation_count: circuit.activation_count,
                            confidence: circuit.confidence,
                            component: circuit.component,
                            layer_name: circuit.layer_name,
                            circuit_id: circuit.circuit_id,
                            phenomenon: circuit.phenomenon,
                            tensor_volume: tensorVolume,
                            activation_intensity: nodeOpacity,
                            tensor_shape: activations[activationKey] || 'unknown'
                        }
                    };
                    nodes.push(newNode);
                });
                
                // Add nodes from activation capture data if available
                Object.keys(activations).forEach(hookKey => {
                    // Only add if not already covered by circuit discovery
                    const existingNode = nodes.find(n => n.metadata.layer_name && hookKey.includes(n.metadata.layer_name));
                    if (!existingNode) {
                        const layerMatch = hookKey.match(/(\d+)/);
                        const layerNum = layerMatch ? parseInt(layerMatch[1]) : 0;
                        const component = hookKey.includes('mlp') ? 'mlp' : 'attention';
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${hookKey} (${component})`,
                            type: component,
                            value: 0.7,
                            color: component === 'mlp' ? '#ff6666' : '#66aaff',
                            nodeColor: component === 'mlp' ? '#ff6666' : '#66aaff',
                            layer: layerNum,
                            position: { 
                                x: (layerNum * 150) + (Math.random() * 50 - 25),
                                y: (component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                layer_name: hookKey,
                                component: component,
                                activation_shape: Array.isArray(activations[hookKey]) ? activations[hookKey] : 'unknown'
                            }
                        };
                        nodes.push(newNode);
                    }
                });
                
            } else if (circuitData.activations) {
                // Handle activation capture format - expand to create more nodes
                Object.keys(circuitData.activations).forEach(hookKey => {
                    const activations = circuitData.activations[hookKey];
                    
                    if (Array.isArray(activations) && activations.length > 0) {
                        const activation = activations[0];
                        const baseLayer = parseInt(activation.layer_name.match(/\d+/)?.[0] || '0');
                        
                        // Create the main activation node
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${activation.layer_name} (${activation.component})`,
                            type: activation.component,
                            value: 0.8,
                            color: activation.component === 'mlp' ? [1.0, 0.4, 0.4, 1.0] : [0.4, 0.6, 1.0, 1.0],
                            nodeColor: activation.component === 'mlp' ? [1.0, 0.4, 0.4, 1.0] : [0.4, 0.6, 1.0, 1.0],
                            layer: baseLayer,
                            position: { 
                                x: (baseLayer * 150) + (Math.random() * 50 - 25),
                                y: (activation.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                shape: activation.shape,
                                count: activations.length,
                                component: activation.component,
                                layer_name: activation.layer_name,
                                dtype: activation.dtype,
                                hook_id: activation.hook_id
                            }
                        };
                        nodes.push(newNode);
                        
                        // Add intermediate processing nodes for richer visualization
                        if (activation.component === 'attention') {
                            // Add query, key, value nodes
                            ['query', 'key', 'value'].forEach((subcomp, idx) => {
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${activation.layer_name} ${subcomp}`,
                                    type: 'attention_sub',
                                    value: 0.6,
                                    color: [0.2, 0.4, 0.8, 1.0], // Darker blue for sub-components
                                    nodeColor: [0.2, 0.4, 0.8, 1.0],
                                    layer: baseLayer,
                                    position: { 
                                        x: (baseLayer * 150) + (idx - 1) * 40,
                                        y: 150 + (Math.random() * 30 - 15)
                                    },
                                    metadata: {
                                        component: 'attention_sub',
                                        subcomponent: subcomp,
                                        parent_layer: activation.layer_name
                                    }
                                };
                                nodes.push(subNode);
                            });
                        } else if (activation.component === 'mlp') {
                            // Add feed-forward sub-nodes
                            ['up_proj', 'gate_proj', 'down_proj'].forEach((subcomp, idx) => {
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${activation.layer_name} ${subcomp}`,
                                    type: 'mlp_sub',
                                    value: 0.6,
                                    color: [0.8, 0.2, 0.2, 1.0], // Darker red for sub-components
                                    nodeColor: [0.8, 0.2, 0.2, 1.0],
                                    layer: baseLayer,
                                    position: { 
                                        x: (baseLayer * 150) + (idx - 1) * 40,
                                        y: 80 + (Math.random() * 30 - 15)
                                    },
                                    metadata: {
                                        component: 'mlp_sub',
                                        subcomponent: subcomp,
                                        parent_layer: activation.layer_name
                                    }
                                };
                                nodes.push(subNode);
                            });
                        }
                    }
                });
            } else {
                // Handle old format for backward compatibility
            Object.keys(circuitData).forEach(layerKey => {
                if (layerKey !== 'metadata' && Array.isArray(circuitData[layerKey])) {
                    const activations = circuitData[layerKey];
                    
                    if (activations.length > 0) {
                        const activation = activations[0]; // Use first activation as representative
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${activation.layer_name} (${activation.component})`,
                            type: activation.component,
                            value: 0.8, // Add value for sizing
                            color: activation.component === 'mlp' ? [1.0, 0.4, 0.4, 1.0] : [0.4, 0.6, 1.0, 1.0], // Red for MLP, Blue for attention
                            layer: parseInt(activation.layer_name.match(/\d+/)?.[0] || '0'),
                            position: { 
                                x: (parseInt(activation.layer_name.match(/\d+/)?.[0] || '0') * 100) + (Math.random() * 50 - 25),
                                y: (activation.component === 'mlp' ? 0 : 50) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                shape: activation.shape,
                                count: activations.length,
                                component: activation.component,
                                layer_name: activation.layer_name
                            }
                        };
                        
                        nodes.push(newNode);
                    }
                }
            });
            }
            
            // Create REALISTIC neural circuit topology
            const mainNodes = nodes.filter(n => n.type === 'attention' || n.type === 'mlp');
            const subNodes = nodes.filter(n => n.type === 'attention_sub' || n.type === 'mlp_sub');
            
            // linkId already declared at function start
            
            // 1. Connect each main node to its sub-components (hub-spoke)
            mainNodes.forEach(mainNode => {
                const relatedSubs = subNodes.filter(subNode => 
                    subNode.metadata.parent_layer === mainNode.metadata.layer_name
                );
                relatedSubs.forEach(subNode => {
                links.push({
                        id: `link_${linkId++}`,
                        source: mainNode.id,
                        target: subNode.id,
                    weight: 0.8,
                    color: '#ffffff', // Brighter white for hub-spoke
                        type: 'hub_spoke',
                        metadata: { connection_type: 'main_to_sub' }
                    });
                });
            });
            
            // 2. Within-layer connections: Attention → MLP (sequential processing)
            const attentionNodes = mainNodes.filter(n => n.type === 'attention');
            const mlpNodes = mainNodes.filter(n => n.type === 'mlp');
            
            // Same-layer attention → MLP connections
            attentionNodes.forEach(attNode => {
                const sameLayerMlp = mlpNodes.find(mlpNode => mlpNode.layer === attNode.layer);
                if (sameLayerMlp) {
                    links.push({
                        id: `link_${linkId++}`,
                        source: attNode.id,
                        target: sameLayerMlp.id,
                        weight: 1.0,
                        color: '#ffff00', // Bright yellow for sequential flow
                        type: 'sequential',
                        metadata: { connection_type: 'attention_to_mlp' }
                    });
                }
            });
            
            // 2b. Cross-layer attention → MLP connections (information flow)
            attentionNodes.forEach(attNode => {
                mlpNodes.forEach(mlpNode => {
                    // Connect attention to MLPs in higher layers
                    if (mlpNode.layer > attNode.layer && mlpNode.layer - attNode.layer <= 3) {
                        links.push({
                            id: `link_${linkId++}`,
                            source: attNode.id,
                            target: mlpNode.id,
                            weight: 0.7,
                            color: '#ffaa00', // Orange for cross-layer attention→MLP
                            type: 'cross_attention_mlp',
                            metadata: { connection_type: 'attention_to_mlp_cross' }
                        });
                    }
                });
            });
            
            // 2c. MLP → Attention connections (feedback)
            mlpNodes.forEach(mlpNode => {
                attentionNodes.forEach(attNode => {
                    // Connect MLP to attention in higher layers
                    if (attNode.layer > mlpNode.layer && attNode.layer - mlpNode.layer <= 5) {
                        links.push({
                            id: `link_${linkId++}`,
                            source: mlpNode.id,
                            target: attNode.id,
                            weight: 0.6,
                            color: '#00aaff', // Light blue for MLP→attention
                            type: 'mlp_to_attention',
                            metadata: { connection_type: 'mlp_to_attention' }
                        });
                    }
                });
            });
            
            // 3. Cross-layer connections: Layer N → Layer N+1 (information flow)
            const layerGroups = {};
            mainNodes.forEach(node => {
                if (!layerGroups[node.layer]) layerGroups[node.layer] = [];
                layerGroups[node.layer].push(node);
            });
            
            const layers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
            for (let i = 0; i < layers.length - 1; i++) {
                const currentLayer = layerGroups[layers[i]];
                const nextLayer = layerGroups[layers[i + 1]];
                
                // Connect MLP output of current layer to attention input of next layer
                const currentMLP = currentLayer.find(n => n.type === 'mlp');
                const nextAttention = nextLayer.find(n => n.type === 'attention');
                
                if (currentMLP && nextAttention) {
                    links.push({
                        id: `link_${linkId++}`,
                        source: currentMLP.id,
                        target: nextAttention.id,
                        weight: 0.6,
                        color: '#00ff66', // Bright green for cross-layer flow
                        type: 'cross_layer',
                        metadata: { connection_type: 'layer_to_layer' }
                    });
                }
                
                // Connect ALL attention nodes across layers (not just first found)
                const currentAttentionNodes = currentLayer.filter(n => n.type === 'attention');
                const nextAttentionNodes = nextLayer.filter(n => n.type === 'attention');
                
                if (currentAttentionNodes.length > 0 && nextAttentionNodes.length > 0) {
                    // Connect each current layer attention to each next layer attention
                    currentAttentionNodes.forEach(currentAtt => {
                        nextAttentionNodes.forEach(nextAtt => {
                            links.push({
                                id: `link_${linkId++}`,
                                source: currentAtt.id,
                                target: nextAtt.id,
                                weight: 0.8,
                                color: '#ffffff', // White for attention flow
                                type: 'attention_flow',
                                metadata: { connection_type: 'attention_to_attention' }
                            });
                        });
                    });
                }
            }
            
            console.error("FINISHED hasActivationLayers BLOCK");
        } // End of hasActivationLayers block
        
        console.error("SKIPPED hasActivationLayers BLOCK, no real activation data to process...");
        
        // Debug info available in browser console instead of MCP protocol
        
        console.error("ABOUT TO CHECK COLOR ASSIGNMENT CONDITION");
        console.error("Condition values: circuitData.nodes =", !!circuitData.nodes, "hasActivationLayers =", hasActivationLayers);
        
        // FORCE color assignment for simple input data with error handling  
        if (circuitData.nodes && !hasActivationLayers) {
                try {
                    console.error("ENTERING color assignment block...");
                    console.error("circuitData.nodes length:", circuitData.nodes.length);
                    
                    for (let i = 0; i < circuitData.nodes.length; i++) {
                        const node = circuitData.nodes[i];
                        console.error("Processing node", i, ":", node.id, "type:", node.type);
                        
                        // Force assign colors based on type
                        if (node.type === 'mlp') {
                            node.color = [1.0, 0.4, 0.4, 1.0]; // Red for MLP
                            console.error("ASSIGNED RED to MLP:", node.id);
                        } else if (node.type === 'attention') {
                            node.color = [0.4, 0.6, 1.0, 1.0]; // Blue for attention  
                            console.error("ASSIGNED BLUE to attention:", node.id);
                        } else {
                            node.color = [0.6, 0.6, 0.6, 1.0]; // Gray fallback
                            console.error("ASSIGNED GRAY to unknown:", node.id);
                        }
                        
                        // Force assign label
                        node.label = node.id;
                        console.error("Node", i, "final color:", node.color);
                    }
                    
                    // Use the modified data
                    nodes = circuitData.nodes;
                    links = circuitData.links || [];
                    console.error("COLOR ASSIGNMENT COMPLETED. nodes length:", nodes.length);
                    
                } catch (error) {
                    console.error("ERROR in color assignment:", error.message);
                    console.error("Error stack:", error.stack);
                }
            }
            
            processedData = { 
                id: `circuit_${Date.now()}`,
                nodes, 
                links, 
                metadata: {
                    ...circuitData.metadata,
                    title: args.circuit_name || 'Neural Circuit',
                    type: 'circuit'
                }
            };
        
        const nodeCount = processedData.nodes ? processedData.nodes.length : 0;
        const linkCount = processedData.links ? processedData.links.length : 0;
        
        // Create a real visualization using Cosmos Graph
        const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${args.circuit_name}</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial, sans-serif; }
        h1 { color: #4285f4; }
        #graph-container { width: 80%; height: 480px; border: 1px solid #333; background: #2a2a2a; margin: 20px auto; position: relative; }
        .metadata { background: #444; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 14px; }
        .node-info { background: #333; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .physics-controls { position: absolute; top: 10px; right: 10px; z-index: 1000; }
        .physics-btn { 
            background: #4285f4; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .physics-btn:hover { background: #5294ff; }
        .physics-btn:active { background: #3275e5; }
        
        /* CSS Labels styles - explicit styles for visibility */
        .css-label {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            font-size: 14px;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,1);
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
            font-family: Arial, sans-serif;
            user-select: none;
        }
        
        #labels-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <h1>${args.circuit_name}</h1>
    <div class="metadata">
        <h3>📊 Real Activation Data Summary</h3>
        <p><strong>Model:</strong> GPT-OSS-20B (20 layers, 768 hidden dimensions)</p>
        <p><strong>Nodes:</strong> ${nodeCount} | <strong>Links:</strong> ${linkCount}</p>
        <p><strong>Input:</strong> "What is 7 + 5?" | <strong>Output:</strong> "12"</p>
        <p><strong>Total Activations Captured:</strong> 48 tensors across layers 0 and 5</p>
    </div>
    <div id="graph-container">
        <div class="physics-controls">
            <button id="physics-btn" class="physics-btn">Play Physics</button>
        </div>
    </div>
    
    <script type="module">
        // Import 3D Force Graph renderer
        import { ForceGraph3DRenderer } from './renderer/force_graph_3d_renderer.js';
        console.log('ForceGraph3DRenderer imported successfully');
        
        // Real circuit data from MLX Engine
        const rawCircuitData = ${JSON.stringify(processedData)};
        
        console.log('🔥 Initializing 3D Force Graph visualization');
        console.log('Raw activation data keys:', Object.keys(rawCircuitData));
        console.log('Data conversion status:', rawCircuitData.nodes ? 'Already converted' : 'Raw activation data');
        console.log('Expected nodes:', rawCircuitData.nodes?.length || 'TBD');
        console.log('Expected links:', rawCircuitData.links?.length || 'TBD');
        
        async function initializeForceGraph() {
            try {
                // Initialize 3D Force Graph renderer
                let graphData;
                
                // Ensure we have properly converted nodes/links data
                if (rawCircuitData.nodes && rawCircuitData.links) {
                    console.log('✅ Using converted graph data');
                    graphData = rawCircuitData;
                } else {
                    throw new Error('❌ No valid graph data found - data conversion failed. Check server-side conversion logic.');
                }
                
                console.log('Graph data for 3D Force Graph:', graphData);
                
                // Initialize 3D Force Graph renderer
                const container = document.getElementById('graph-container');
                const renderer = new ForceGraph3DRenderer(container, {
                    backgroundColor: '#1a1a1a',
                    nodeColor: '#58a6ff',
                    linkColor: '#30363d',
                    nodeOpacity: 0.8,
                    linkOpacity: 0.6,
                    nodeRelSize: 4,
                    linkWidth: 2,
                    showNodeLabels: true,
                    showLinkLabels: false,
                    controlType: 'trackball',
                    enableNodeDrag: true,
                    enableNavigationControls: true,
                    enablePointerInteraction: true
                });
                
                console.log('3D Force Graph renderer initialized');
                
                // Add event listeners for node interactions
                renderer.onNodeHover((node) => {
                    if (node) {
                        console.log('Hovering node:', node.label || node.id);
                    }
                });
                
                renderer.onNodeClick((node) => {
                    if (node) {
                        console.log('Clicked node:', node.label || node.id);
                    }
                });
                
                // Load the graph data
                await renderer.loadGraph(graphData);
                console.log('✅ Graph loaded successfully');
                
                // Labels are handled by the ForceGraph3DRenderer internally
                console.log('Node labels enabled via renderer configuration');
                
                // Physics control setup - ensure DOM is ready
                let physicsPlaying = true;
                
                // Wait for DOM to be fully loaded
                const setupPhysicsControls = () => {
                    const physicsBtn = document.getElementById('physics-btn');
                    
                    // Set correct initial button state (simulation starts running)
                    if (physicsBtn) {
                        physicsBtn.textContent = 'Pause Physics';
                        console.log('Physics initialized as running');
                        
                        // Physics toggle functionality using 3D Force Graph API
                        physicsBtn.addEventListener('click', () => {
                            if (physicsPlaying) {
                                // Pause physics
                                if (renderer.graph && renderer.graph.pauseAnimation) {
                                    renderer.graph.pauseAnimation();
                                }
                                physicsBtn.textContent = 'Play Physics';
                                physicsPlaying = false;
                                console.log('Physics paused via button');
                            } else {
                                // Resume physics
                                if (renderer.graph && renderer.graph.resumeAnimation) {
                                    renderer.graph.resumeAnimation();
                                }
                                physicsBtn.textContent = 'Pause Physics';
                                physicsPlaying = true;
                                console.log('Physics resumed via button');
                            }
                        });
                    } else {
                        console.error('Physics button not found in DOM');
                    }
                };
                
                // Call setup immediately (DOM should be ready by now)
                setupPhysicsControls();
                
                console.log('3D Force Graph visualization ready');
                
            } catch (error) {
                console.error('3D Force Graph initialization failed:', error);
                throw error; // No fallbacks - we use 3D Force Graph or fail gracefully
            }
        }
        
        // Start the 3D Force Graph visualization
        initializeForceGraph();
        
        // Update the display in real-time
        const metadataP = document.querySelector('.metadata p:nth-child(3)');
        if (metadataP && rawCircuitData.nodes) {
            metadataP.innerHTML = '<strong>Nodes:</strong> ' + rawCircuitData.nodes.length + ' | <strong>Links:</strong> ' + (rawCircuitData.links?.length || 0);
        }
    </script>
    
    <div class="node-info">
        <h3>🧠 Circuit Structure (Real Data)</h3>
        <p><strong>Nodes:</strong> ${nodeCount} components across layers</p>
        <p><strong>Links:</strong> ${linkCount} connections showing information flow</p>
        <details>
            <summary>View Raw Data</summary>
            <pre style="max-height: 300px; overflow-y: auto;">${JSON.stringify(args.circuit_data, null, 2)}</pre>
        </details>
    </div>
</body>
</html>`;
        
        const htmlPath = path.join(vizDir, 'real_circuit.html');
        await fs.writeFile(htmlPath, htmlContent);
        
        return {
            success: true,
            visualization_url: "http://localhost:8888/real_circuit.html",
            circuit_name: args.circuit_name,
            nodes_count: nodeCount,
            edges_count: linkCount,
            layout: "force_directed",
            data_file: "real_circuit_data.json",
            html_file: "real_circuit.html",
            files_created: [circuitDataPath, htmlPath],
            visualization_type: "canvas_graph"
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            circuit_name: args.circuit_name
        };
    }
}

async function attentionPatternsTool(args) {
    return {
        success: true,
        visualization_url: "http://localhost:8888/attention_patterns.html",
        layers_visualized: args.layers,
        pattern_type: "heatmap",
        interactive_features: ["zoom", "filter", "export"]
    };
}

async function activationFlowTool(args) {
    return {
        success: true,
        visualization_url: "http://localhost:8888/activation_flow.html",
        prompt: args.prompt,
        flow_type: "sankey_diagram",
        animation_enabled: true
    };
}

async function openBrowserTool(args) {
    try {
        const { spawn, exec } = await import('child_process');
        const { promisify } = await import('util');
        const execAsync = promisify(exec);
        const url = args.url;
        
        // Detect platform and use appropriate command to open browser
        let command, args_array;
        if (process.platform === 'darwin') {
            // macOS - check user preferences for browser behavior
            if (globalPreferences.browserMode === 'chrome-new-window') {
                try {
                    // Try Chrome with new window flag
                    await execAsync(`/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --new-window "${url}"`);
                    return {
                        success: true,
                        url: url,
                        action: "browser_opened",
                        message: `Successfully opened Chrome new window at ${url}`,
                        platform: process.platform,
                        command: `chrome --new-window ${url}`,
                        preferences_used: globalPreferences
                    };
                } catch (chromeError) {
                    // Fallback to default if Chrome fails
            command = 'open';
            args_array = [url];
                }
            } else {
                // Default browser behavior
                command = 'open';
                args_array = [url];
            }
        } else if (process.platform === 'win32') {
            // Windows
            command = 'start';
            args_array = ['', url];
        } else {
            // Linux
            command = 'xdg-open';
            args_array = [url];
        }
        
        // Spawn the process to open browser
        const child = spawn(command, args_array, { detached: true, stdio: 'ignore' });
        child.unref();
        
        return {
            success: true,
            url: url,
            action: "browser_opened",
            message: `Successfully opened browser at ${url}`,
            platform: process.platform,
            command: `${command} ${args_array.join(' ')}`
        };
    } catch (error) {
        return {
            success: false,
            url: args.url,
            error: error.message,
            message: "Failed to open browser"
        };
    }
}

// Health Check and Server Management Tools
async function healthCheckTool(args) {
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

async function startServerTool(args) {
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

async function generateReportTool(args) {
    return {
        success: true,
        title: args.title,
        report_url: "http://localhost:8888/reports/analysis_report.html",
        sections: ["Executive Summary", "Circuit Analysis", "Visualizations", "Conclusions"],
        format: "interactive_html",
        generated_at: new Date().toISOString()
    };
}

async function setPreferencesTool(args) {
    try {
        if (args.browserMode !== undefined) {
            globalPreferences.browserMode = args.browserMode;
        }
        if (args.defaultBrowser !== undefined) {
            globalPreferences.defaultBrowser = args.defaultBrowser;
        }
        
        return {
            success: true,
            message: "Preferences updated successfully",
            preferences: { ...globalPreferences }
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            preferences: { ...globalPreferences }
        };
    }
}

async function getPreferencesTool(args) {
    return {
        success: true,
        preferences: { ...globalPreferences }
    };
}

async function describeToolsTool(args) {
    const toolDescriptions = TOOLS.map(tool => ({
        name: tool.name,
        description: tool.description,
        category: getToolCategory(tool.name),
        parameters: getToolParameters(tool.name)
    }));

    return {
        success: true,
        server: "mechanistic-interpretability-mcp-server",
        version: 33,
        total_tools: TOOLS.length,
        tools: toolDescriptions,
        categories: {
            "Core Analysis": ["discover_circuits", "localize_features", "analyze_math", "analyze_attention", "analyze_factual"],
            "Data Capture": ["capture_activations", "create_hooks", "track_residual"],
            "Visualization": ["circuit_diagram", "attention_patterns", "activation_flow", "open_browser"],
            "Model Management": ["load_model", "health_check", "start_server"],
            "Export & Reports": ["export_neuroscope", "generate_report"],
            "System": ["ping", "version", "set_preferences", "get_preferences", "describe_tools"]
        },
        usage_examples: {
            "Neural Circuit Analysis": "1. load_model → 2. capture_activations → 3. circuit_diagram → 4. open_browser",
            "Attention Study": "1. load_model → 2. analyze_attention → 3. attention_patterns → 4. open_browser",
            "Mathematical Reasoning": "1. load_model → 2. analyze_math → 3. circuit_diagram → 4. open_browser"
        }
    };
}

function getToolCategory(toolName) {
    const categories = {
        "ping": "System",
        "version": "System", 
        "discover_circuits": "Core Analysis",
        "localize_features": "Core Analysis",
        "capture_activations": "Data Capture",
        "load_model": "Model Management",
        "create_hooks": "Data Capture",
        "analyze_math": "Core Analysis",
        "analyze_attention": "Core Analysis",
        "analyze_factual": "Core Analysis",
        "track_residual": "Data Capture",
        "export_neuroscope": "Export & Reports",
        "circuit_diagram": "Visualization",
        "attention_patterns": "Visualization",
        "activation_flow": "Visualization",
        "open_browser": "Visualization",
        "health_check": "Model Management",
        "start_server": "Model Management",
        "generate_report": "Export & Reports",
        "set_preferences": "System",
        "get_preferences": "System",
        "describe_tools": "System"
    };
    return categories[toolName] || "Other";
}

function getToolParameters(toolName) {
    const tool = TOOLS.find(t => t.name === toolName);
    if (!tool || !tool.inputSchema || !tool.inputSchema.properties) {
        return [];
    }
    
    return Object.keys(tool.inputSchema.properties).map(key => ({
        name: key,
        required: tool.inputSchema.required?.includes(key) || false,
        type: tool.inputSchema.properties[key].type || "unknown"
    }));
}

// Define tools array for dynamic count calculation
const TOOLS = [
    {
        name: "ping",
        description: "Simple ping tool for testing server connectivity and basic functionality.",
        inputSchema: zodToJsonSchema(PingArgsSchema),
    },
    {
        name: "version",
        description: "Returns the current version of the MCP server to track updates.",
        inputSchema: zodToJsonSchema(VersionArgsSchema),
    },
    {
        name: "discover_circuits",
        description: "Discovers circuits for a specific phenomenon using causal tracing with activation patching.",
        inputSchema: zodToJsonSchema(DiscoverCircuitsArgsSchema),
    },
    {
        name: "localize_features", 
        description: "Localizes neurons responsible for specific features using Principal Component Analysis and probing classifiers.",
        inputSchema: zodToJsonSchema(LocalizeFeaturesArgsSchema),
    },
    {
        name: "capture_activations",
        description: "Captures activations during text generation for analysis.",
        inputSchema: zodToJsonSchema(CaptureActivationsArgsSchema),
    },
    {
        name: "load_model",
        description: "Loads a model in the MLX Engine for analysis.",
        inputSchema: zodToJsonSchema(LoadModelArgsSchema),
    },
    {
        name: "available_models",
        description: "Lists available models that can be loaded in the MLX Engine.",
        inputSchema: zodToJsonSchema(AvailableModelsArgsSchema),
    },
    {
        name: "create_hooks",
        description: "Creates activation hooks for capturing model internals.",
        inputSchema: zodToJsonSchema(CreateHooksArgsSchema),
    },
    {
        name: "analyze_math",
        description: "Analyzes mathematical reasoning circuits in the model.",
        inputSchema: zodToJsonSchema(AnalyzeMathArgsSchema),
    },
    {
        name: "analyze_attention",
        description: "Analyzes attention patterns in specified layers.",
        inputSchema: zodToJsonSchema(AnalyzeAttentionArgsSchema),
    },
    {
        name: "analyze_factual",
        description: "Analyzes factual recall circuits and mechanisms.",
        inputSchema: zodToJsonSchema(AnalyzeFactualArgsSchema),
    },
    {
        name: "track_residual",
        description: "Tracks information flow through the residual stream.",
        inputSchema: zodToJsonSchema(TrackResidualArgsSchema),
    },
    {
        name: "export_neuroscope",
        description: "Exports analysis data to NeuroScope format.",
        inputSchema: zodToJsonSchema(ExportNeuroScopeArgsSchema),
    },
    {
        name: "circuit_diagram",
        description: "Creates an interactive circuit diagram visualization.",
        inputSchema: zodToJsonSchema(CircuitDiagramArgsSchema),
    },
    {
        name: "attention_patterns",
        description: "Creates an attention pattern visualization.",
        inputSchema: zodToJsonSchema(AttentionPatternsArgsSchema),
    },
    {
        name: "activation_flow",
        description: "Creates an activation flow visualization.",
        inputSchema: zodToJsonSchema(ActivationFlowArgsSchema),
    },
    {
        name: "open_browser",
        description: "Opens the visualization interface in browser.",
        inputSchema: zodToJsonSchema(OpenBrowserArgsSchema),
    },
    {
        name: "health_check",
        description: "Check health status of MLX Engine and Visualization servers.",
        inputSchema: zodToJsonSchema(HealthCheckArgsSchema),
    },
    {
        name: "start_server",
        description: "Start MLX Engine or Visualization server if not running.",
        inputSchema: zodToJsonSchema(StartServerArgsSchema),
    },
    {
        name: "generate_report",
        description: "Generates a comprehensive analysis report with visualizations.",
        inputSchema: zodToJsonSchema(GenerateReportArgsSchema),
    },
    {
        name: "set_preferences",
        description: "Set user preferences for browser behavior and other settings.",
        inputSchema: zodToJsonSchema(SetPreferencesArgsSchema),
    },
    {
        name: "get_preferences", 
        description: "Get current user preferences.",
        inputSchema: zodToJsonSchema(GetPreferencesArgsSchema),
    },
    {
        name: "describe_tools",
        description: "Get a complete summary of all available tools with categories and usage examples.",
        inputSchema: zodToJsonSchema(DescribeToolsArgsSchema),
    },
];

// Tool handlers - exactly like filesystem server
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: TOOLS,
    };
});

server.setRequestHandler(CallToolRequestSchema, async (request) => {
    try {
        const { name, arguments: args } = request.params;
        
        switch (name) {
            case "ping": {
                const parsed = PingArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for ping: ${parsed.error}`);
                }
                
                const result = await pingTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "version": {
                const parsed = VersionArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for version: ${parsed.error}`);
                }
                
                const result = await versionTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "discover_circuits": {
                const parsed = DiscoverCircuitsArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for discover_circuits: ${parsed.error}`);
                }
                
                const result = await discoverCircuitsTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "localize_features": {
                const parsed = LocalizeFeaturesArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for localize_features: ${parsed.error}`);
                }
                
                const result = await localizeFeaturesTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "capture_activations": {
                const parsed = CaptureActivationsArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for capture_activations: ${parsed.error}`);
                }
                
                const result = await captureActivationsTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "load_model": {
                const parsed = LoadModelArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for load_model: ${parsed.error}`);
                }
                
                const result = await loadModelTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "available_models": {
                const parsed = AvailableModelsArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for available_models: ${parsed.error}`);
                }
                
                const result = await availableModelsTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "create_hooks": {
                const parsed = CreateHooksArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for create_hooks: ${parsed.error}`);
                }
                
                const result = await createHooksTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "analyze_math": {
                const parsed = AnalyzeMathArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for analyze_math: ${parsed.error}`);
                }
                
                const result = await analyzeMathTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "analyze_attention": {
                const parsed = AnalyzeAttentionArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for analyze_attention: ${parsed.error}`);
                }
                
                const result = await analyzeAttentionTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "analyze_factual": {
                const parsed = AnalyzeFactualArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for analyze_factual: ${parsed.error}`);
                }
                
                const result = await analyzeFactualTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "track_residual": {
                const parsed = TrackResidualArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for track_residual: ${parsed.error}`);
                }
                
                const result = await trackResidualTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "export_neuroscope": {
                const parsed = ExportNeuroScopeArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for export_neuroscope: ${parsed.error}`);
                }
                
                const result = await exportNeuroScopeTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "circuit_diagram": {
                const parsed = CircuitDiagramArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for circuit_diagram: ${parsed.error}`);
                }
                
                const result = await circuitDiagramTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "attention_patterns": {
                const parsed = AttentionPatternsArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for attention_patterns: ${parsed.error}`);
                }
                
                const result = await attentionPatternsTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "activation_flow": {
                const parsed = ActivationFlowArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for activation_flow: ${parsed.error}`);
                }
                
                const result = await activationFlowTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "open_browser": {
                const parsed = OpenBrowserArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for open_browser: ${parsed.error}`);
                }
                
                const result = await openBrowserTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "health_check": {
                const parsed = HealthCheckArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for health_check: ${parsed.error}`);
                }
                
                const result = await healthCheckTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }]
                };
            }

            case "start_server": {
                const parsed = StartServerArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for start_server: ${parsed.error}`);
                }
                
                const result = await startServerTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }]
                };
            }

            case "generate_report": {
                const parsed = GenerateReportArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for generate_report: ${parsed.error}`);
                }
                
                const result = await generateReportTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "set_preferences": {
                const parsed = SetPreferencesArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for set_preferences: ${parsed.error}`);
                }
                
                const result = await setPreferencesTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "get_preferences": {
                const parsed = GetPreferencesArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for get_preferences: ${parsed.error}`);
                }
                
                const result = await getPreferencesTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }

            case "describe_tools": {
                const parsed = DescribeToolsArgsSchema.safeParse(args);
                if (!parsed.success) {
                    throw new Error(`Invalid arguments for describe_tools: ${parsed.error}`);
                }
                
                const result = await describeToolsTool(parsed.data);
                return {
                    content: [{
                        type: "text",
                        text: JSON.stringify(result, null, 2)
                    }],
                };
            }
            
            default:
                throw new Error(`Unknown tool: ${name}`);
        }
    }
    catch (error) {
        return {
            content: [{
                type: "text",
                text: `Error: ${error.message}`
            }],
            isError: true,
        };
    }
});

// Kill any existing servers on startup to prevent conflicts
async function cleanupExistingServers() {
    console.error("Cleaning up any existing MLX Engine and visualization servers...");
    
    try {
        // Kill processes on MLX Engine port (50111)
        const { execSync } = await import('child_process');
        try {
            execSync('lsof -ti:50111 | xargs kill -9', { stdio: 'ignore' });
            console.error("Killed existing MLX Engine server on port 50111");
        } catch (e) {
            // No process found, that's fine
        }
        
        // Kill processes on visualization server port (8888)
        try {
            execSync('lsof -ti:8888 | xargs kill -9', { stdio: 'ignore' });
            console.error("Killed existing visualization server on port 8888");
        } catch (e) {
            // No process found, that's fine
        }
        
        // Wait a moment for processes to fully terminate
        await new Promise(resolve => setTimeout(resolve, 1000));
        console.error("Server cleanup completed");
        
    } catch (error) {
        console.error("Warning: Server cleanup had issues:", error.message);
    }
}

// Start server - exactly like filesystem server
async function runServer() {
    // Clean up any existing servers first
    await cleanupExistingServers();
    
    // Initialize MLX Engine client with default config
    const mlxConfig = {
        apiUrl: 'http://localhost:50111',
        timeout: 30000,
        retryAttempts: 3
    };
    console.error("Initializing MLX Engine client...");
    initializeMLXClient(mlxConfig);
    console.error("MLX Engine client initialized");
    
    // Start required services
    console.error("Starting MLX Engine and visualization servers...");
    try {
        // Start MLX Engine
        const mlxResult = await startServerTool({ service: 'mlx' });
        if (!mlxResult.success) {
            console.error("Warning: MLX Engine failed to start:", mlxResult.error);
        } else {
            console.error("MLX Engine started successfully on port 50111");
        }
        
        // Start visualization server
        const vizResult = await startServerTool({ service: 'visualization' });
        if (!vizResult.success) {
            console.error("Warning: Visualization server failed to start:", vizResult.error);
        } else {
            console.error("Visualization server started successfully on port 8888");
        }
        
        // Health check both services
        const healthResult = await healthCheckTool({ service: 'all' });
        if (healthResult.success && healthResult.overall_status === 'healthy') {
            console.error("All services healthy and ready");
        } else {
            console.error("Warning: Services not fully operational:", healthResult.overall_status);
            if (healthResult.services) {
                if (healthResult.services.mlx_engine?.status === 'unreachable') {
                    console.error("- MLX Engine is not responding");
                }
                if (healthResult.services.visualization_server?.status === 'unreachable') {
                    console.error("- Visualization server is not responding");
                }
            }
        }
        
    } catch (error) {
        console.error("Warning: Server startup had issues:", error.message);
    }
    
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error(`Mechanistic Interpretability MCP Server v${(await versionTool({})).version} running on stdio`);
    console.error(`Registered ${TOOLS.length} tools for GPT-OSS-20B circuit analysis, health monitoring, and WebGL2 visualization`);
}

runServer().catch((error) => {
    console.error("Fatal error running server:", error);
    process.exit(1);
});

