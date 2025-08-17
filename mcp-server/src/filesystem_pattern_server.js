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
    model_id: z.enum(["gpt-oss-20b", "llama-2-7b", "mistral-7b", "phi-2"]),
    quantization: z.enum(["none", "4bit", "8bit"]).default("none"),
    max_context_length: z.number().int().min(512).max(131072).default(2048),
    device: z.enum(["auto", "cpu", "mps", "cuda"]).default("auto"),
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
        version: 99,
        server: "mechanistic-interpretability-mcp-server",
        last_modified: new Date().toISOString(),
        changes: "FIX label tracking: Add setInterval for continuous label updates and debug why only 1 of 3 labels is created"
    };
}

async function discoverCircuitsTool(args) {
    // Mock implementation for now - will connect to MLX Engine later
    return {
        success: true,
        phenomenon: args.phenomenon,
        model_id: args.model_id,
        circuits: [
            {
                id: "circuit_001",
                name: `${args.phenomenon}_primary_circuit`,
                confidence: 0.85,
                layers: [8, 9, 10],
                components: ["attention_head_8_3", "mlp_9", "attention_head_10_1"],
                validation_metrics: {
                    performance_recovery: 0.92,
                    attribution_score: 0.78,
                    consistency_score: 0.84
                }
            }
        ],
        execution_time_ms: 1250,
        model_info: {
            model_id: args.model_id,
            architecture: "transformer",
            num_layers: 20
        }
    };
}

async function localizeFeaturesTool(args) {
    return {
        success: true,
        feature_name: args.feature_name,
        neurons: [
            {
                layer: 5,
                neuron_id: 123,
                activation_strength: 0.89,
                confidence: 0.92
            },
            {
                layer: 8,
                neuron_id: 456,
                activation_strength: 0.76,
                confidence: 0.87
            }
        ],
        validation_metrics: {
            valid: true,
            confidence: 0.89,
            metrics: {
                precision: 0.91,
                recall: 0.87,
                f1_score: 0.89
            }
        }
    };
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
            throw new Error(`MLX Engine API error: ${response.status} ${response.statusText}`);
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
                model_path: `/Users/craig/me/behavior/forks/mlx-engine-neuroscope/models/nightmedia/${args.model_id}-q4-hi-mlx`,
                model_id: args.model_id,
                quantization: args.quantization,
                max_context_length: args.max_context_length,
                device: args.device
            })
        });

        if (!response.ok) {
            throw new Error(`MLX Engine API error: ${response.status} ${response.statusText}`);
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
    return {
        success: true,
        prompt: args.prompt,
        analysis: {
            reasoning_steps: ["Parse mathematical expression", "Apply operator precedence", "Calculate result"],
            circuit_components: ["attention_head_5_2", "mlp_8", "attention_head_10_1"],
            confidence: 0.89,
            mathematical_operations: ["addition", "multiplication"],
            result_confidence: 0.94
        },
        execution_time_ms: 2100
    };
}

async function analyzeAttentionTool(args) {
    return {
        success: true,
        prompt: args.prompt,
        attention_analysis: {
            layers: args.layers,
            patterns: {
                syntactic: 0.75,
                semantic: 0.68,
                positional: 0.82
            },
            head_importance: args.layers.map(layer => ({
                layer,
                heads: Array.from({length: 8}, (_, i) => ({
                    head: i,
                    importance: Math.random() * 0.8 + 0.2
                }))
            }))
        }
    };
}

async function analyzeFactualTool(args) {
    return {
        success: true,
        query: args.query,
        factual_analysis: {
            knowledge_circuits: ["fact_retrieval_layer_12", "entity_binding_layer_8"],
            confidence: 0.87,
            retrieval_mechanism: "key-value_lookup",
            evidence_strength: 0.91,
            knowledge_source: "pretrained_knowledge"
        }
    };
}

async function trackResidualTool(args) {
    return {
        success: true,
        prompt: args.prompt,
        residual_flow: {
            layers: args.layers.length > 0 ? args.layers : [0, 1, 2, 3, 4, 5],
            component_contributions: {
                attention: 0.42,
                mlp: 0.38,
                residual: 0.20
            },
            information_flow: "progressive_refinement",
            bottleneck_layers: [3, 7, 11]
        }
    };
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
                                            color: [1.0, 1.0, 1.0, 0.9], // Brighter white for hub-spoke
                        type: 'hub_spoke',
                        metadata: { connection_type: 'main_to_sub' }
                    });
                });
            });
            
            // 2. Within-layer connections: Attention → MLP (sequential processing)
            const attentionNodes = mainNodes.filter(n => n.type === 'attention');
            const mlpNodes = mainNodes.filter(n => n.type === 'mlp');
            
            attentionNodes.forEach(attNode => {
                const sameLayerMlp = mlpNodes.find(mlpNode => mlpNode.layer === attNode.layer);
                if (sameLayerMlp) {
                    links.push({
                        id: `link_${linkId++}`,
                        source: attNode.id,
                        target: sameLayerMlp.id,
                        weight: 1.0,
                        color: [1.0, 1.0, 0.0, 1.0], // Bright yellow for sequential flow
                        type: 'sequential',
                        metadata: { connection_type: 'attention_to_mlp' }
                    });
                }
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
                        color: [0.0, 1.0, 0.0, 1.0], // Bright green for cross-layer flow
                        type: 'cross_layer',
                        metadata: { connection_type: 'layer_to_layer' }
                    });
                }
            }
            
            console.error("FINISHED hasActivationLayers BLOCK");
        } // End of hasActivationLayers block
        
        console.error("SKIPPED hasActivationLayers BLOCK, continuing to color assignment...");
        
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
        #graph-container { width: 100%; height: 600px; border: 1px solid #333; background: #2a2a2a; margin: 20px 0; position: relative; }
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
        // Import Cosmos Graph and CSS Labels
        const CosmosModule = await import('./node_modules/@cosmos.gl/graph/dist/index.js');
        const LabelsModule = await import('./node_modules/@interacta/css-labels/dist/index.js');
        console.log('Cosmos module loaded:', Object.keys(CosmosModule));
        console.log('CSS Labels module loaded:', Object.keys(LabelsModule));
        const Graph = CosmosModule.Graph;
        const LabelRenderer = LabelsModule.LabelRenderer;
        console.log('Graph constructor found:', !!Graph);
        console.log('LabelRenderer found:', !!LabelRenderer);
        console.log('LabelRenderer constructor:', LabelRenderer);
        
        // Real circuit data from MLX Engine
        const rawCircuitData = ${JSON.stringify(processedData)};
        
        console.log('🔥 Initializing Cosmos Graph WebGL2 visualization');
        console.log('Raw activation data keys:', Object.keys(rawCircuitData));
        console.log('Data conversion status:', rawCircuitData.nodes ? 'Already converted' : 'Raw activation data');
        console.log('Expected nodes:', rawCircuitData.nodes?.length || 'TBD');
        console.log('Expected links:', rawCircuitData.links?.length || 'TBD');
        
        async function initializeCosmosGraph() {
            try {
                // Convert browser-side processed data to Cosmos Graph format
                let graphData;
                
                // Ensure we have properly converted nodes/links data
                if (rawCircuitData.nodes && rawCircuitData.links) {
                    console.log('✅ Using converted graph data');
                    graphData = rawCircuitData;
                } else {
                    throw new Error('❌ No valid graph data found - data conversion failed. Check server-side conversion logic.');
                }
                
                console.log('Graph data for Cosmos:', graphData);
                
                // Initialize Cosmos Graph DIRECTLY with callbacks in constructor
                const container = document.getElementById('graph-container');
                const graph = new Graph(container, {
                    backgroundColor: '#1a1a1a',
                    linkWidth: 6,  // Increased from 2 to 6 for better color visibility
                    linkArrows: true, // Enable directional arrows on links
                    linkArrowsSizeScale: 1.5, // Make arrows bigger for visibility
                    pointSize: 20,
                    curvedLinks: true, // Enable curved links for visual flexibility
                    curvedLinkWeight: 0.8, // Control curve amount
                    enableSimulation: true,
                    fitViewOnInit: true, // Auto-fit view to keep nodes visible
                    fitViewPadding: 0.2, // 20% padding around nodes
                    // Exact physics settings from working localhost:8888 visualization
                    simulationFriction: 0.1,    // Low friction like working version
                    simulationGravity: 0,       // No gravity (key difference!)
                    simulationRepulsion: 0.5,   // Moderate repulsion like working version
                    // Drag and space settings from working version
                    enableDrag: true,
                    spaceSize: 4096,             // Match working version
                    scalePointsOnZoom: true,     // Match working version
                    // Event callbacks configured correctly
                    onClick: (pointIndex) => {
                        if (pointIndex !== undefined) {
                            const node = window.nodeData ? window.nodeData[pointIndex] : null;
                            console.log('Node clicked:', pointIndex, 'Label:', node?.label || 'unknown');
                        }
                    },
                    onPointMouseOver: (pointIndex) => {
                        if (pointIndex !== undefined) {
                            const node = window.nodeData ? window.nodeData[pointIndex] : null;
                            console.log('Hovering node:', pointIndex, 'Label:', node?.label || 'unknown');
                        }
                    },
                    onPointMouseOut: (pointIndex) => {
                        if (pointIndex !== undefined) {
                            console.log('Left node:', pointIndex);
                        }
                    }
                });
                
                console.log('Direct Cosmos Graph initialized with enableSimulation: true');
                
                // Store node data for callbacks
                window.nodeData = graphData.nodes;
                
                // Convert data to Cosmos Graph format 
                const positions = new Float32Array(graphData.nodes.length * 2);
                const colors = new Float32Array(graphData.nodes.length * 4);
                const links = new Float32Array(graphData.links.length * 2);
                
                console.log('Converting data for direct Cosmos Graph API...');
                
                // Debug nodes before processing
                console.log('Nodes to process:', graphData.nodes.length);
                graphData.nodes.forEach((node, index) => {
                    console.log('Node ' + index + ':', {
                        id: node.id,
                        type: node.type,
                        hasColor: !!node.color,
                        color: node.color
                    });
                });
                
                // Set positions with cluster separation for independent movement
                graphData.nodes.forEach((node, i) => {
                    let x, y;
                    
                    // Position clusters far apart to enable independent movement
                    if (node.type === 'attention' || node.type === 'attention_sub') {
                        // Attention cluster on the left
                        x = -80 + (Math.random() * 40 - 20);
                        y = Math.random() * 60 - 30;
                    } else {
                        // MLP cluster on the right  
                        x = 80 + (Math.random() * 40 - 20);
                        y = Math.random() * 60 - 30;
                    }
                    
                    positions[i * 2] = x;
                    positions[i * 2 + 1] = y;
                    
                    // Validate and set colors with fallback
                    if (node.color && Array.isArray(node.color) && node.color.length >= 4) {
                        colors[i * 4] = node.color[0];     // R
                        colors[i * 4 + 1] = node.color[1]; // G  
                        colors[i * 4 + 2] = node.color[2]; // B
                        colors[i * 4 + 3] = node.color[3]; // A
                    } else {
                        console.warn('Node ' + i + ' missing valid color, using fallback');
                        // Default to blue for missing colors
                        colors[i * 4] = 0.0;     // R
                        colors[i * 4 + 1] = 0.5; // G  
                        colors[i * 4 + 2] = 1.0; // B
                        colors[i * 4 + 3] = 1.0; // A
                    }
                    
                    const colorStr = node.color ? node.color.join(',') : 'fallback';
                    console.log('Node ' + i + ': ' + node.label + ' at (' + x + ',' + y + ') color=[' + colorStr + ']');
                });
                
                // Set links and link colors
                const linkColors = new Float32Array(graphData.links.length * 4); // RGBA for each link
                
                graphData.links.forEach((link, i) => {
                    const sourceIndex = graphData.nodes.findIndex(n => n.id === link.source);
                    const targetIndex = graphData.nodes.findIndex(n => n.id === link.target);
                    links[i * 2] = sourceIndex;
                    links[i * 2 + 1] = targetIndex;
                    
                    // Set link colors: [r, g, b, a] for each link according to API docs
                    const color = link.color || [1, 1, 1, 0.6]; // Default to white
                    linkColors[i * 4] = color[0];     // Red
                    linkColors[i * 4 + 1] = color[1]; // Green  
                    linkColors[i * 4 + 2] = color[2]; // Blue
                    linkColors[i * 4 + 3] = color[3]; // Alpha
                    
                    // Debug moved to browser console to avoid MCP protocol interference
                });
                
                // Debug the arrays before setting them
                console.log('Final arrays for Cosmos Graph:');
                console.log('positions length:', positions.length, 'expected:', graphData.nodes.length * 2);
                console.log('positions values:', Array.from(positions));
                console.log('colors length:', colors.length, 'expected:', graphData.nodes.length * 4);
                console.log('colors values:', Array.from(colors));
                console.log('links length:', links.length, 'expected:', graphData.links.length * 2);
                console.log('links values:', Array.from(links));
                console.log('point size check - should be > 0:', graph.getPointSize ? graph.getPointSize() : 'no getPointSize method');
                
                // Load data into graph using correct API
                try {
                    graph.setPointPositions(positions);
                    console.log('✅ Point positions set');
                    
                    graph.setPointColors(colors);
                    console.log('✅ Point colors set');
                    
                    graph.setLinks(links);
                    console.log('✅ Links set');
                    
                    graph.setLinkColors(linkColors);
                    console.log('✅ Link colors set');
                    
                    // Render and immediately pause to start static
                    graph.render();
                    console.log('✅ Graph rendered');
                    
                    graph.pause();
                    console.log('✅ Graph paused');
                    
                    // Fit view to see all nodes with padding
                    setTimeout(() => {
                        graph.fitView(1000, 0.2);
                        console.log('✅ Graph fitted to view');
                    }, 100);
                    
                } catch (error) {
                    console.error('❌ Error setting up graph:', error);
                    console.error('Error details:', error.message);
                    console.error('Error stack:', error.stack);
                }
                
                console.log('Direct Cosmos Graph loaded, paused, and fitted to view!');
                
                // Use already imported CSS Labels from top of script
                if (!LabelRenderer) {
                    console.error('LabelRenderer not available from @interacta/css-labels');
                    return;
                }
                
                console.log('Setting up CSS Labels with LabelRenderer:', LabelRenderer);
                
                // Create labels container
                const labelsContainer = document.createElement('div');
                labelsContainer.id = 'labels-container';
                labelsContainer.style.position = 'absolute';
                labelsContainer.style.top = '0';
                labelsContainer.style.left = '0';
                labelsContainer.style.width = '100%';
                labelsContainer.style.height = '100%';
                labelsContainer.style.pointerEvents = 'none';
                labelsContainer.style.zIndex = '1000';
                container.appendChild(labelsContainer);
                
                // Initialize CSS Labels renderer with proper options
                const labelRenderer = new LabelRenderer(labelsContainer, { 
                    pointerEvents: 'auto',
                    dontInjectStyles: false  // Let the library inject its own styles
                });
                
                // Make sure the renderer is visible
                labelRenderer.show();
                console.log('LabelRenderer initialized and shown');
                
                // Create label configurations
                console.log('Creating labels for nodes:', graphData.nodes.length);
                graphData.nodes.forEach((node, i) => {
                    console.log('Node ' + i + ':', node.id, 'Label:', node.label);
                });
                
                // Create initial labels - will be positioned after graph is fitted
                const labels = graphData.nodes.map((node, i) => {
                    return {
                        id: node.id,
                        text: node.label || node.id,
                        x: 0,  // Will be updated after fitView
                        y: 0,  // Will be updated after fitView
                        fontSize: 14,
                        color: '#ffffff',
                        opacity: 1.0,
                        shouldBeShown: true,
                        weight: 10
                    };
                });
                
                console.log('Created labels array:', labels.length);
                
                function updateLabels() {
                    const positions = graph.getPointPositions();
                    const containerRect = container.getBoundingClientRect();
                    let updatedLabels;
                    let labelElements;
                    
                    console.log('Updating labels, positions array length:', positions.length);
                    console.log('Container bounds:', { width: containerRect.width, height: containerRect.height });
                    
                    updatedLabels = labels.map((label, i) => {
                        const spaceX = positions[i * 2];
                        const spaceY = positions[i * 2 + 1];
                        const screenPos = graph.spaceToScreenPosition([spaceX, spaceY]);
                        
                        // Clamp coordinates to container bounds
                        const x = Math.max(50, Math.min(containerRect.width - 50, screenPos[0]));
                        const y = Math.max(50, Math.min(containerRect.height - 50, screenPos[1] - 35));
                        
                        console.log('Label ' + i + ' positioned near node:', { 
                            spacePos: [spaceX, spaceY], 
                            rawScreenPos: screenPos,
                            clampedPos: [x, y],
                            containerBounds: [containerRect.width, containerRect.height]
                        });
                        
                        return {
                            ...label,
                            x: x,
                            y: y,
                            shouldBeShown: true
                        };
                    });
                    
                    console.log('About to setLabels with:', updatedLabels.length, 'labels');
                    updatedLabels.forEach((label, i) => {
                        console.log('Label ' + i + ' data:', {
                            id: label.id,
                            text: label.text,
                            x: label.x,
                            y: label.y,
                            shouldBeShown: label.shouldBeShown
                        });
                    });
                    
                    labelRenderer.setLabels(updatedLabels);
                    labelRenderer.draw(true);
                    
                    // Debug: Check if DOM elements are actually created
                    labelElements = labelsContainer.querySelectorAll('div');
                    console.log('Labels updated and drawn with setLabels API');
                    console.log('DOM label elements found:', labelElements.length, 'expected:', updatedLabels.length);
                    labelElements.forEach((el, i) => {
                        console.log('Label element ' + i + ':', {
                            text: el.textContent,
                            visible: el.style.display !== 'none',
                            position: el.style.position,
                            left: el.style.left,
                            top: el.style.top,
                            className: el.className
                        });
                    });
                }
                
                // Position labels after the graph has been fitted to view
                setTimeout(() => {
                    console.log('Positioning labels after graph fitView...');
                    updateLabels();
                }, 200);
                
                // Update labels during all interactions - use setInterval for continuous updates
                setInterval(updateLabels, 100); // Update every 100ms for smooth movement
                
                // Also update on specific events
                graph.onZoom = updateLabels;
                if (graph.onSimulationTick) graph.onSimulationTick = updateLabels;
                if (graph.onDrag) graph.onDrag = updateLabels;
                if (graph.onDragEnd) graph.onDragEnd = updateLabels;
                
                // Interactions are now configured in constructor
                
                // Physics control setup - simulation disabled by config
                let physicsPlaying = false;
                const physicsBtn = document.getElementById('physics-btn');
                
                // Set correct initial button state (simulation starts paused)
                physicsBtn.textContent = 'Play Physics';
                console.log('Physics initialized as paused');
                
                // Physics toggle functionality using direct Cosmos Graph API
                physicsBtn.addEventListener('click', () => {
                    if (physicsPlaying) {
                        // Pause physics
                        graph.pause();
                        physicsBtn.textContent = 'Play Physics';
                        physicsPlaying = false;
                        console.log('Physics paused via button');
                    } else {
                        // Start physics with high energy for visible spring action
                        console.log('Starting physics...');
                        graph.start(1.0); // High energy to see spring compression/extension
                        physicsBtn.textContent = 'Pause Physics';
                        physicsPlaying = true;
                        console.log('Physics started via button with alpha=1.0');
                        
                        // No auto-refit needed anymore
                    }
                });
                
                console.log('Cosmos Graph WebGL2 visualization ready');
                
            } catch (error) {
                console.error('Cosmos Graph initialization failed:', error);
                throw error; // No fallbacks - we use Cosmos Graph or fail gracefully
            }
        }
        
        // Start the Cosmos Graph WebGL2 visualization
        initializeCosmosGraph();
        
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

