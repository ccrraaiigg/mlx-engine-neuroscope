import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
// Define paths
const baseDir = path.join(__dirname, 'src/visualization');
const rendererDir = path.join(baseDir, 'renderer');

// Log paths for debugging
console.log('Base directory:', baseDir);
console.log('Renderer directory:', rendererDir);

// Serve static files from the visualization directory
app.use(express.static(baseDir));

// Serve the renderer directory
app.use('/renderer', express.static(rendererDir, {
    setHeaders: (res, path) => {
        console.log('Serving file:', path);
    }
}));

// Serve node_modules from the project root
app.use('/node_modules', express.static(path.join(__dirname, 'node_modules'), {
    setHeaders: (res, path) => {
        // Set CORS headers for all node_modules files
        res.setHeader('Access-Control-Allow-Origin', '*');
        // Set appropriate content type for .js files
        if (path.endsWith('.js')) {
            res.setHeader('Content-Type', 'application/javascript');
        }
    }
}));

// Log all requests for debugging
app.use((req, res, next) => {
  console.log(`[${new Date().toISOString()}] ${req.method} ${req.path}`);
  next();
});

// Debug route to check file existence
app.get('/check-renderer', (req, res) => {
    const fs = require('fs');
    const rendererPath = path.join(rendererDir, 'cosmos_graph_renderer.js');
    const exists = fs.existsSync(rendererPath);
    res.json({
        rendererPath,
        exists,
        files: fs.readdirSync(rendererDir)
    });
});

// Redirect root to the test page
app.get('/', (req, res) => {
  res.redirect('/test_browser.html');
});

// Alias /test to the test page
app.get('/test', (req, res) => {
  res.redirect('/test_browser.html');
});

// Start the server
app.listen(PORT, () => {
  console.log(`🌐 Visualization server running at http://localhost:${PORT}`);
  console.log(`📊 Open http://localhost:${PORT} in your browser`);
});