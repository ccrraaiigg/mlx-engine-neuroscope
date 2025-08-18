const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const port = 8888;
const pidFile = path.join(__dirname, 'visualization_server.pid');

const mimeTypes = {
    '.html': 'text/html',
    '.js': 'application/javascript',
    '.mjs': 'application/javascript',
    '.css': 'text/css',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.wav': 'audio/wav',
    '.mp4': 'video/mp4',
    '.woff': 'application/font-woff',
    '.ttf': 'application/font-ttf',
    '.eot': 'application/vnd.ms-fontobject',
    '.otf': 'application/font-otf',
    '.wasm': 'application/wasm'
};

function checkExistingInstance() {
    if (fs.existsSync(pidFile)) {
        try {
            const oldPid = parseInt(fs.readFileSync(pidFile, 'utf8').trim());
            
            // Check if the process is still running
            try {
                process.kill(oldPid, 0); // Signal 0 just checks if process exists
                console.log(`Visualization server already running with PID ${oldPid}`);
                console.log('Exiting to prevent multiple instances.');
                process.exit(1);
            } catch (e) {
                // Process not running, remove stale PID file
                console.log(`Removing stale PID file for PID ${oldPid}`);
                fs.unlinkSync(pidFile);
            }
        } catch (e) {
            // Invalid PID file, remove it
            if (fs.existsSync(pidFile)) {
                fs.unlinkSync(pidFile);
            }
        }
    }
}

function writePidFile() {
    fs.writeFileSync(pidFile, process.pid.toString());
    console.log(`PID file written: ${pidFile} (PID: ${process.pid})`);
}

function removePidFile() {
    if (fs.existsSync(pidFile)) {
        fs.unlinkSync(pidFile);
        console.log('PID file removed');
    }
}

function signalHandler(signal) {
    console.log(`Received signal ${signal}`);
    removePidFile();
    console.log('Visualization server shutting down...');
    process.exit(0);
}

const server = http.createServer((req, res) => {
    const parsedUrl = url.parse(req.url);
    let pathname = `.${parsedUrl.pathname}`;
    
    // Health check endpoint
    if (parsedUrl.pathname === '/health') {
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({
            status: 'healthy',
            service: 'mlx-engine-neuroscope',
            component: 'Visualization Web Server',
            version: '1.1.0',
            timestamp: new Date().toISOString(),
            port: port,
            pid: process.pid,
            pid_file: pidFile,
            ready: true
        }));
        return;
    }
    
    // Default to index.html
    if (pathname === './') {
        pathname = './index.html';
    }
    
    const ext = path.parse(pathname).ext;
    const mimeType = mimeTypes[ext] || 'text/plain';
    
    fs.readFile(pathname, (err, data) => {
        if (err) {
            res.writeHead(404);
            res.end(`File ${pathname} not found!`);
            return;
        }
        
        res.writeHead(200, {
            'Content-Type': mimeType,
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type'
        });
        res.end(data);
    });
});

// Check for existing instance before starting
checkExistingInstance();

// Set up signal handlers for graceful shutdown
process.on('SIGINT', () => signalHandler('SIGINT'));
process.on('SIGTERM', () => signalHandler('SIGTERM'));

// Write PID file
writePidFile();

server.listen(port, () => {
    console.log(`Visualization server running at http://localhost:${port}/`);
    console.log(`PID: ${process.pid}`);
    console.log('Press Ctrl+C to stop the server');
});