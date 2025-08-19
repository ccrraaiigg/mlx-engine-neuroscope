import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema definition
export const DescribeToolsArgsSchema = z.object({
    random_string: z.string().optional(),
});

// Tool definition
export const describeToolsToolDefinition = {
    name: "describe_tools",
    description: "Get a complete summary of all available tools with categories and usage examples.",
    inputSchema: zodToJsonSchema(DescribeToolsArgsSchema),
};

// Helper functions
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

function getToolParameters(toolName, tools) {
    const tool = tools.find(t => t.name === toolName);
    if (!tool || !tool.inputSchema || !tool.inputSchema.properties) {
        return [];
    }
    
    return Object.keys(tool.inputSchema.properties).map(key => ({
        name: key,
        required: tool.inputSchema.required?.includes(key) || false,
        type: tool.inputSchema.properties[key].type || "unknown"
    }));
}

// Tool implementation
export async function describeToolsTool(args, tools) {
    // If tools array is not provided, we'll return a basic response
    if (!tools || !Array.isArray(tools)) {
        return {
            success: false,
            error: "Tools array not available",
            server: "mechanistic-interpretability-mcp-server"
        };
    }

    const toolDescriptions = tools.map(tool => ({
        name: tool.name,
        description: tool.description,
        category: getToolCategory(tool.name),
        parameters: getToolParameters(tool.name, tools)
    }));

    return {
        success: true,
        server: "mechanistic-interpretability-mcp-server",
        total_tools: tools.length,
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