import { pingToolDefinition, pingTool, PingArgsSchema } from './ping.js';
import { versionToolDefinition, versionTool, VersionArgsSchema } from './version.js';
import { discoverCircuitsToolDefinition, discoverCircuitsTool, DiscoverCircuitsArgsSchema } from './discover_circuits.js';
import { localizeFeaturesToolDefinition, localizeFeaturesTool, LocalizeFeaturesArgsSchema } from './localize_features.js';
import { captureActivationsToolDefinition, captureActivationsTool, CaptureActivationsArgsSchema } from './capture_activations.js';
import { loadModelToolDefinition, loadModelTool, LoadModelArgsSchema } from './load_model.js';
import { availableModelsToolDefinition, availableModelsTool, AvailableModelsArgsSchema } from './available_models.js';
import { createHooksToolDefinition, createHooksTool, CreateHooksArgsSchema } from './create_hooks.js';
import { analyzeMathTool, AnalyzeMathArgsSchema } from './analyze_math.js';
import { analyzeAttentionTool, AnalyzeAttentionArgsSchema } from './analyze_attention.js';
import { analyzeFactualTool, AnalyzeFactualArgsSchema } from './analyze_factual.js';
import { trackResidualTool, TrackResidualArgsSchema } from './track_residual.js';
import { exportNeuroScopeTool, ExportNeuroScopeArgsSchema } from './export_neuroscope.js';
import { circuitDiagramTool, CircuitDiagramArgsSchema } from './circuit_diagram.js';
import { attentionPatternsTool, AttentionPatternsArgsSchema } from './attention_patterns.js';
import { activationFlowTool, ActivationFlowArgsSchema } from './activation_flow.js';
import { openBrowserTool, OpenBrowserArgsSchema } from './open_browser.js';
import { healthCheckToolDefinition, healthCheckTool, HealthCheckArgsSchema } from './health_check.js';
import { startServerToolDefinition, startServerTool, StartServerArgsSchema } from './start_server.js';
import { generateReportToolDefinition, generateReportTool, GenerateReportArgsSchema } from './generate_report.js';
import { setPreferencesToolDefinition, setPreferencesTool, SetPreferencesArgsSchema } from './set_preferences.js';
import { getPreferencesToolDefinition, getPreferencesTool, GetPreferencesArgsSchema } from './get_preferences.js';
import { describeToolsToolDefinition, describeToolsTool, DescribeToolsArgsSchema } from './describe_tools.js';
import { zodToJsonSchema } from "zod-to-json-schema";

// Create tool definitions for extracted tools
const analyzeMathToolDefinition = {
    name: "analyze_math",
    description: "Analyzes mathematical reasoning circuits in the model.",
    inputSchema: zodToJsonSchema(AnalyzeMathArgsSchema),
};

const analyzeAttentionToolDefinition = {
    name: "analyze_attention",
    description: "Analyzes attention patterns in specified layers.",
    inputSchema: zodToJsonSchema(AnalyzeAttentionArgsSchema),
};

const analyzeFactualToolDefinition = {
    name: "analyze_factual",
    description: "Analyzes factual recall circuits and mechanisms.",
    inputSchema: zodToJsonSchema(AnalyzeFactualArgsSchema),
};

const trackResidualToolDefinition = {
    name: "track_residual",
    description: "Tracks information flow through the residual stream.",
    inputSchema: zodToJsonSchema(TrackResidualArgsSchema),
};

const exportNeuroScopeToolDefinition = {
    name: "export_neuroscope",
    description: "Exports analysis data to NeuroScope format.",
    inputSchema: zodToJsonSchema(ExportNeuroScopeArgsSchema),
};

const circuitDiagramToolDefinition = {
    name: "circuit_diagram",
    description: "Creates an interactive circuit diagram visualization.",
    inputSchema: zodToJsonSchema(CircuitDiagramArgsSchema),
};

const attentionPatternsToolDefinition = {
    name: "attention_patterns",
    description: "Creates an attention pattern visualization.",
    inputSchema: zodToJsonSchema(AttentionPatternsArgsSchema),
};

const activationFlowToolDefinition = {
    name: "activation_flow",
    description: "Creates an activation flow visualization.",
    inputSchema: zodToJsonSchema(ActivationFlowArgsSchema),
};

const openBrowserToolDefinition = {
    name: "open_browser",
    description: "Opens the visualization interface in browser.",
    inputSchema: zodToJsonSchema(OpenBrowserArgsSchema),
};

// Export all tool definitions (excluding open_browser - kept for internal use only)
export const TOOL_DEFINITIONS = [
    pingToolDefinition,
    versionToolDefinition,
    discoverCircuitsToolDefinition,
    localizeFeaturesToolDefinition,
    captureActivationsToolDefinition,
    loadModelToolDefinition,
    availableModelsToolDefinition,
    createHooksToolDefinition,
    analyzeMathToolDefinition,
    analyzeAttentionToolDefinition,
    analyzeFactualToolDefinition,
    trackResidualToolDefinition,
    exportNeuroScopeToolDefinition,
    circuitDiagramToolDefinition,
    attentionPatternsToolDefinition,
    activationFlowToolDefinition,
    // openBrowserToolDefinition, // Commented out - tool kept for internal use only
    healthCheckToolDefinition,
    startServerToolDefinition,
    generateReportToolDefinition,
    setPreferencesToolDefinition,
    getPreferencesToolDefinition,
    describeToolsToolDefinition,
];

// Export all tool handlers
export const TOOL_HANDLERS = {
    ping: pingTool,
    version: versionTool,
    discover_circuits: discoverCircuitsTool,
    localize_features: localizeFeaturesTool,
    capture_activations: captureActivationsTool,
    load_model: loadModelTool,
    available_models: availableModelsTool,
    create_hooks: createHooksTool,
    analyze_math: analyzeMathTool,
    analyze_attention: analyzeAttentionTool,
    analyze_factual: analyzeFactualTool,
    track_residual: trackResidualTool,
    export_neuroscope: exportNeuroScopeTool,
    circuit_diagram: circuitDiagramTool,
    attention_patterns: attentionPatternsTool,
    activation_flow: activationFlowTool,
    open_browser: openBrowserTool,
    health_check: healthCheckTool,
    start_server: startServerTool,
    generate_report: generateReportTool,
    set_preferences: setPreferencesTool,
    get_preferences: getPreferencesTool,
    describe_tools: describeToolsTool,
};

// Export all tool schemas
export const TOOL_SCHEMAS = {
    ping: PingArgsSchema,
    version: VersionArgsSchema,
    discover_circuits: DiscoverCircuitsArgsSchema,
    localize_features: LocalizeFeaturesArgsSchema,
    capture_activations: CaptureActivationsArgsSchema,
    load_model: LoadModelArgsSchema,
    available_models: AvailableModelsArgsSchema,
    create_hooks: CreateHooksArgsSchema,
    analyze_math: AnalyzeMathArgsSchema,
    analyze_attention: AnalyzeAttentionArgsSchema,
    analyze_factual: AnalyzeFactualArgsSchema,
    track_residual: TrackResidualArgsSchema,
    export_neuroscope: ExportNeuroScopeArgsSchema,
    circuit_diagram: CircuitDiagramArgsSchema,
    attention_patterns: AttentionPatternsArgsSchema,
    activation_flow: ActivationFlowArgsSchema,
    open_browser: OpenBrowserArgsSchema,
    health_check: HealthCheckArgsSchema,
    start_server: StartServerArgsSchema,
    generate_report: GenerateReportArgsSchema,
    set_preferences: SetPreferencesArgsSchema,
    get_preferences: GetPreferencesArgsSchema,
    describe_tools: DescribeToolsArgsSchema,
};

// Helper functions
export function getToolHandler(toolName) {
    return TOOL_HANDLERS[toolName];
}

export function getToolSchema(toolName) {
    return TOOL_SCHEMAS[toolName];
}  

export function validateToolArgs(toolName, args) {
    const schema = getToolSchema(toolName);
    if (!schema) {
        throw new Error(`Unknown tool: ${toolName}`);
    }
    return schema.parse(args);
}