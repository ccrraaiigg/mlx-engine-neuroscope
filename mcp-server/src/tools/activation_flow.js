import { z } from "zod";

export const ActivationFlowArgsSchema = z.object({
    activation_data: z.object({}),
    prompt: z.string(),
});

export async function activationFlowTool(args) {
    return {
        success: true,
        visualization_url: "http://localhost:8888/activation_flow.html",
        prompt: args.prompt,
        flow_type: "sankey_diagram",
        animation_enabled: true
    };
}