import { z } from "zod";
import { zodToJsonSchema } from "zod-to-json-schema";

// Global preferences object (shared state)
const globalPreferences = {
    browserMode: 'chrome-new-window', // 'default' or 'chrome-new-window' - DEFAULT: chrome-new-window for better UX
    defaultBrowser: 'system', // 'system', 'chrome', 'safari', 'firefox'
};

// Schema definition
export const SetPreferencesArgsSchema = z.object({
    browserMode: z.enum(['default', 'chrome-new-window']).optional(),
    defaultBrowser: z.enum(['system', 'chrome', 'safari', 'firefox']).optional(),
});

// Tool definition
export const setPreferencesToolDefinition = {
    name: "set_preferences",
    description: "Set user preferences for browser behavior and other settings.",
    inputSchema: zodToJsonSchema(SetPreferencesArgsSchema),
};

// Tool implementation
export async function setPreferencesTool(args) {
    try {
        if (args.browserMode !== undefined) {
            globalPreferences.browserMode = args.browserMode;
        }
        if (args.defaultBrowser !== undefined) {
            globalPreferences.defaultBrowser = args.defaultBrowser;
        }
        
        return {
            success: true,
            message: "Preferences updated successfully",
            preferences: { ...globalPreferences }
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            preferences: { ...globalPreferences }
        };
    }
}

// Export preferences for use by other tools
export { globalPreferences };