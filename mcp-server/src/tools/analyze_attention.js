import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const AnalyzeAttentionArgsSchema = z.object({
    prompt: z.string().min(1),
    layers: z.array(z.number().int().min(0).max(50)).min(1).max(10),
});

async function analyzeAttentionTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('../services/mlx_tools.js');
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

export { analyzeAttentionTool, AnalyzeAttentionArgsSchema };

export const analyzeAttention = {
    name: "analyze_attention",
    description: "Analyzes attention patterns in specified layers.",
    inputSchema: zodToJsonSchema(AnalyzeAttentionArgsSchema),
    argsSchema: AnalyzeAttentionArgsSchema,
    handler: analyzeAttentionTool
};