import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const AnalyzeFactualArgsSchema = z.object({
    query: z.string().min(1),
    max_tokens: z.number().int().min(5).max(200).default(50),
    analysis_depth: z.enum(["shallow", "medium", "deep"]).default("medium"),
});

async function analyzeFactualTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('../services/mlx_tools.js');
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

export { analyzeFactualTool, AnalyzeFactualArgsSchema };

export const analyzeFactual = {
    name: "analyze_factual",
    description: "Analyzes factual recall circuits and mechanisms.",
    inputSchema: zodToJsonSchema(AnalyzeFactualArgsSchema),
    argsSchema: AnalyzeFactualArgsSchema,
    handler: analyzeFactualTool
};