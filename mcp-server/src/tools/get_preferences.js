import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";
import { globalPreferences } from "./set_preferences.js";

// Schema definition
export const GetPreferencesArgsSchema = z.object({
    random_string: z.string().optional(),
});

// Tool definition
export const getPreferencesToolDefinition = {
    name: "get_preferences",
    description: "Get current user preferences.",
    inputSchema: zodToJsonSchema(GetPreferencesArgsSchema),
};

// Tool implementation
export async function getPreferencesTool(args) {
    return {
        success: true,
        preferences: { ...globalPreferences }
    };
}