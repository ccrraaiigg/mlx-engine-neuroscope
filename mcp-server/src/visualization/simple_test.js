import { ForceGraph3DRenderer } from './renderer/force_graph_3d_renderer.js';

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM fully loaded, starting application...');
    
    try {
        console.log('Initializing renderer...');
        
        // Create the renderer with enhanced configuration
        const container = document.getElementById('graph-container');
        const renderer = new ForceGraph3DRenderer(container, {
            backgroundColor: '#0d1117',
            nodeColor: '#58a6ff',
            linkColor: '#30363d',
            nodeOpacity: 0.8,
            linkOpacity: 0.6,
            nodeRelSize: 4,
            linkWidth: 1.5,
            showNodeLabels: true,
            showLinkLabels: false,
            controlType: 'trackball',
            enableNodeDrag: true,
            enableNavigationControls: true,
            enablePointerInteraction: true
        });
        
        // Add event listeners
        renderer.onNodeHover((node) => {
            if (node) {
                console.log('Hovered node:', node);
            }
        });
        
        renderer.onNodeClick((node) => {
            if (node) {
                console.log('Clicked node:', node);
            }
        });
        
        // Create sample data for 3D Force Graph
        const nodes = Array(50).fill(0).map((_, i) => {
            const angle = (i / 50) * Math.PI * 2;
            const radius = 300;
            return {
                id: `node-${i}`,
                label: `Node ${i}`,
                value: 2 + Math.random() * 3,
                x: Math.cos(angle) * radius,
                y: Math.sin(angle) * radius,
                z: (Math.random() - 0.5) * 200,
                color: `hsl(${Math.random() * 360}, 70%, 60%)`
            };
        });
        
        // Create links for 3D Force Graph
        const links = [];
        
        // Create a circular structure
        for (let i = 0; i < nodes.length; i++) {
            const next = (i + 1) % nodes.length;
            links.push({
                source: nodes[i].id,
                target: nodes[next].id,
                label: `Link ${i}-${next}`,
                value: 1.0,
                color: '#ffa500'  // Orange links
            });
        }
        
        // Add some random connections
        const numRandomLinks = Math.floor(nodes.length * 0.3);
        for (let i = 0; i < numRandomLinks; i++) {
            const sourceIdx = Math.floor(Math.random() * nodes.length);
            let targetIdx;
            
            do {
                targetIdx = Math.floor(Math.random() * nodes.length);
            } while (targetIdx === sourceIdx);
            
            links.push({
                source: nodes[sourceIdx].id,
                target: nodes[targetIdx].id,
                label: `Random ${i}`,
                value: 0.3 + Math.random() * 0.7,
                color: '#30363d'  // Gray links
            });
        }
        
        // Load the data using the renderer (now fixed)
        await renderer.loadGraph({ nodes, links });
        console.log('Graph loaded successfully');
        
    } catch (error) {
        console.error('Error:', error);
        alert(`Error: ${error.message}\nCheck console for details`);
    }
});
