import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const AnalyzeMathArgsSchema = z.object({
    prompt: z.string().min(1),
    max_tokens: z.number().int().min(10).max(500).default(100),
    temperature: z.number().min(0).max(1).default(0.1),
    analysis_depth: z.enum(["shallow", "medium", "deep"]).default("medium"),
});

async function analyzeMathTool(args) {
    // AGENT.md: Never fake anything. If information is missing, DO NOT guess, "mock", or simulate.
    // Report the exact parameters and error details instead of returning mock data.
    try {
        const mlxTools = await import('../services/mlx_tools.js');
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

export { analyzeMathTool, AnalyzeMathArgsSchema };

export const analyzeMath = {
    name: "analyze_math",
    description: "Analyzes mathematical reasoning circuits in the model.",
    inputSchema: zodToJsonSchema(AnalyzeMathArgsSchema),
    argsSchema: AnalyzeMathArgsSchema,
    handler: analyzeMathTool
};