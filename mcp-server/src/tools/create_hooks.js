import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Schema for create_hooks tool arguments
export const CreateHooksArgsSchema = z.object({
    layers: z.array(z.number().int().min(0).max(50)).min(1),
    hook_type: z.enum(["activation", "gradient", "both"]).default("activation"),
    components: z.array(z.enum(["mlp", "attention", "residual", "all"])).default(["mlp", "attention"]),
    capture_gradients: z.boolean().default(false),
});

// Handler function for create_hooks tool
export async function createHooksTool(args) {
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

// Tool definition for MCP server
export const createHooksToolDefinition = {
    name: "create_hooks",
    description: "Creates activation hooks for capturing model internals.",
    inputSchema: zodToJsonSchema(CreateHooksArgsSchema),
};