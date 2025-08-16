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

// Input schemas using Zod (like filesystem server)
const PingArgsSchema = z.object({
    message: z.string().optional(),
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
    circuit_data: z.object({}),
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

const GenerateReportArgsSchema = z.object({
    title: z.string(),
    analysis_data: z.object({}),
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
    return {
        success: true,
        prompt: args.prompt,
        generated_text: "This is a mock response for testing.",
        activations: {
            attention_patterns: {
                layers: [0, 1, 2, 3, 4],
                heads_per_layer: 8,
                data_format: "matrix"
            },
            residual_stream: args.capture_residual_stream ? {
                layers: [0, 1, 2, 3, 4],
                dimensions: 512
            } : null
        },
        metadata: {
            total_tokens: 25,
            generation_time_ms: 850,
            model_info: {
                name: "test-model",
                layers: 20,
                hidden_size: 512
            }
        }
    };
}

// MLX Engine Tools
async function loadModelTool(args) {
    return {
        success: true,
        model_id: args.model_id,
        status: "loaded",
        model_info: {
            architecture: "transformer",
            num_layers: 20,
            hidden_size: 512,
            vocab_size: 32000,
            quantization: args.quantization,
            device: args.device,
            memory_usage_mb: 8192
        },
        load_time_ms: 15000
    };
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
    return {
        success: true,
        visualization_url: "http://localhost:8888/circuit_diagram.html",
        circuit_name: args.circuit_name,
        nodes_count: 12,
        edges_count: 18,
        layout: "force_directed"
    };
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
    return {
        success: true,
        url: args.url,
        action: "browser_opened",
        message: `Opening visualization at ${args.url}`
    };
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

// Tool handlers - exactly like filesystem server
server.setRequestHandler(ListToolsRequestSchema, async () => {
    return {
        tools: [
            {
                name: "ping",
                description: "Simple ping tool for testing server connectivity and basic functionality.",
                inputSchema: zodToJsonSchema(PingArgsSchema),
            },
            {
                name: "discover_circuits",
                description: "Discovers circuits for a specific phenomenon using causal tracing with activation patching.",
                inputSchema: zodToJsonSchema(DiscoverCircuitsArgsSchema),
            },
            {
                name: "localize_features", 
                description: "Localizes neurons responsible for specific features using PCA and probing classifiers.",
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
                name: "generate_report",
                description: "Generates a comprehensive analysis report with visualizations.",
                inputSchema: zodToJsonSchema(GenerateReportArgsSchema),
            },
        ],
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

// Start server - exactly like filesystem server
async function runServer() {
    const transport = new StdioServerTransport();
    await server.connect(transport);
    console.error("Mechanistic Interpretability MCP Server running on stdio");
    console.error("Registered 16 tools for neural network analysis");
}

runServer().catch((error) => {
    console.error("Fatal error running server:", error);
    process.exit(1);
});

