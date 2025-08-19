import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

const TrackResidualArgsSchema = z.object({
    prompt: z.string().min(1),
    layers: z.array(z.number().int().min(0).max(50)).default([]),
    components: z.array(z.enum(["attention", "mlp", "both"])).default(["both"]),
});

async function trackResidualTool(args) {
    // AGENT.md: Never fake anything. Try to call real MLX implementation.
    try {
        const mlxTools = await import('../services/mlx_tools.js');
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

export { trackResidualTool, TrackResidualArgsSchema };

export const trackResidual = {
    name: "track_residual",
    description: "Tracks information flow through the residual stream.",
    inputSchema: zodToJsonSchema(TrackResidualArgsSchema),
    argsSchema: TrackResidualArgsSchema,
    handler: trackResidualTool
};