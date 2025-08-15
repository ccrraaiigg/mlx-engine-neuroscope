import { CosmosGraphRenderer } from './renderer/cosmos_graph_renderer.js';

// Wait for the DOM to be fully loaded
document.addEventListener('DOMContentLoaded', async () => {
    console.log('DOM fully loaded, starting application...');
    
    try {
        console.log('Initializing renderer...');
        
        // Create the renderer with enhanced configuration
        const container = document.getElementById('graph-container');
        const renderer = new CosmosGraphRenderer(container, {
            width: container.clientWidth,
            height: container.clientHeight,
            backgroundColor: [0.1, 0.1, 0.1, 1],
            // Enhanced rendering settings
            linkWidth: 2,
            linkColor: [1, 1, 1, 0.6],  // Brighter links
            pointColor: [0.2, 0.6, 1, 1],  // Brighter blue nodes
            pointSize: 6,
            // Enable interactions
            enableNodeHover: true,
            enableNodeClick: true,
            // Physics settings
            simulationGravity: 0.1,
            simulationRepulsion: 0.8,
            linkDistance: 100
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
        
        // Create sample data
        const nodes = Array(50).fill(0).map((_, i) => {
            const angle = (i / 50) * Math.PI * 2;
            const radius = 300;
            return {
                id: `node-${i}`,
                value: 2 + Math.random() * 3,
                position: {
                    x: Math.cos(angle) * radius,
                    y: Math.sin(angle) * radius
                },
                size: 8 + Math.random() * 4,
                color: [
                    0.2 + Math.random() * 0.8,
                    0.2 + Math.random() * 0.6,
                    0.8 + Math.random() * 0.2,
                    0.9
                ]
            };
        });
        
        // Create links
        const links = [];
        
        // Create a circular structure
        for (let i = 0; i < nodes.length; i++) {
            const next = (i + 1) % nodes.length;
            links.push({
                source: nodes[i].id,
                target: nodes[next].id,
                id: `link-${i}-${next}`,
                weight: 1.0,
                value: 1.0,
                width: 2,
                color: [1.0, 0.8, 0.2, 0.8]  // Yellow links
            });
        }
        
        // Add some random connections
        const numRandomLinks = Math.floor(nodes.length * 0.8);
        for (let i = 0; i < numRandomLinks; i++) {
            const sourceIdx = Math.floor(Math.random() * nodes.length);
            let targetIdx;
            
            do {
                targetIdx = Math.floor(Math.random() * nodes.length);
            } while (targetIdx === sourceIdx);
            
            links.push({
                source: nodes[sourceIdx].id,
                target: nodes[targetIdx].id,
                id: `rlink-${i}`,
                weight: 0.3 + Math.random() * 0.7,
                value: 0.3 + Math.random() * 0.7,
                width: 1,
                color: [1.0, 1.0, 1.0, 0.4]  // Semi-transparent white
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
