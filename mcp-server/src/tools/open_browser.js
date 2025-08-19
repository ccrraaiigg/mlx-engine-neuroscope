import { z } from "zod";

export const OpenBrowserArgsSchema = z.object({
    url: z.string().default("http://localhost:8888"),
});

export async function openBrowserTool(args) {
    try {
        const { spawn, exec } = await import('child_process');
        const { promisify } = await import('util');
        const execAsync = promisify(exec);
        const url = args.url;
        
        // Global preferences for browser behavior
        const globalPreferences = {
            browserMode: 'chrome-new-window', // 'default' or 'chrome-new-window' - DEFAULT: chrome-new-window for better UX
            defaultBrowser: 'system', // 'system', 'chrome', 'safari', 'firefox'
        };
        
        // Detect platform and use appropriate command to open browser
        let command, args_array;
        if (process.platform === 'darwin') {
            // macOS - check user preferences for browser behavior
            if (globalPreferences.browserMode === 'chrome-new-window') {
                try {
                    // Try Chrome with new window flag
                    await execAsync(`/Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome --new-window "${url}"`);
                    return {
                        success: true,
                        url: url,
                        action: "browser_opened",
                        message: `Successfully opened Chrome new window at ${url}`,
                        platform: process.platform,
                        command: `chrome --new-window ${url}`,
                        preferences_used: globalPreferences
                    };
                } catch (chromeError) {
                    // Fallback to default if Chrome fails
            command = 'open';
            args_array = [url];
                }
            } else {
                // Default browser behavior
                command = 'open';
                args_array = [url];
            }
        } else if (process.platform === 'win32') {
            // Windows
            command = 'start';
            args_array = ['', url];
        } else {
            // Linux
            command = 'xdg-open';
            args_array = [url];
        }
        
        // Spawn the process to open browser
        const child = spawn(command, args_array, { detached: true, stdio: 'ignore' });
        child.unref();
        
        return {
            success: true,
            url: url,
            action: "browser_opened",
            message: `Successfully opened browser at ${url}`,
            platform: process.platform,
            command: `${command} ${args_array.join(' ')}`
        };
    } catch (error) {
        return {
            success: false,
            url: args.url,
            error: error.message,
            message: "Failed to open browser"
        };
    }
}