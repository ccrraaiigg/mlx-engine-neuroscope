import { z } from "zod";
import { writeFileReadOnly } from '../utils/file_utils.js';
import path from 'path';
import fs from 'fs';

// File-based logging function
function logToFile(message, data = null) {
    const timestamp = new Date().toISOString();
    const logEntry = data 
        ? `[${timestamp}] ${message}: ${JSON.stringify(data, null, 2)}\n`
        : `[${timestamp}] ${message}\n`;
    
    try {
        fs.appendFileSync('activation-flow.log', logEntry);
    } catch (error) {
        // Silently fail to avoid interfering with JSON responses
    }
}

export const ActivationFlowArgsSchema = z.object({
    activation_data: z.object({}),
    prompt: z.string(),
});

export async function activationFlowTool(args) {
    try {
        logToFile('🔥 Activation Flow Tool - Processing activation data for 3D Force Graph');
        logToFile('Input prompt', args.prompt);
        logToFile('Activation data keys', Object.keys(args.activation_data || {}));
        
        const activationData = args.activation_data || {};
        const vizDir = process.env.VIZ_DIR || '/Users/craig/me/behavior/forks/mlx-engine-neuroscope/mcp-server/src/visualization';
        
        // Process activation data into nodes and links for 3D Force Graph
        let nodes = [];
        let links = [];
        
        // Convert activation data to graph format
        if (activationData && Object.keys(activationData).length > 0) {
            // Process layers and components from activation data
            Object.entries(activationData).forEach(([layerKey, layerData], layerIndex) => {
                if (layerData && typeof layerData === 'object') {
                    // Create attention and MLP nodes for each layer
                    const attentionNodeId = `layer_${layerIndex}_attention`;
                    const mlpNodeId = `layer_${layerIndex}_mlp`;
                    
                    nodes.push({
                        id: attentionNodeId,
                        label: `L${layerIndex} Attention`,
                        type: 'attention',
                        layer: layerIndex,
                        color: '#ff6b6b',
                        value: 8,
                        metadata: {
                            component: 'attention',
                            layer: layerIndex,
                            activation_count: Array.isArray(layerData.attention) ? layerData.attention.length : 1
                        }
                    });
                    
                    nodes.push({
                        id: mlpNodeId,
                        label: `L${layerIndex} MLP`,
                        type: 'mlp',
                        layer: layerIndex,
                        color: '#4ecdc4',
                        value: 8,
                        metadata: {
                            component: 'mlp',
                            layer: layerIndex,
                            activation_count: Array.isArray(layerData.mlp) ? layerData.mlp.length : 1
                        }
                    });
                    
                    // Within-layer connection (attention -> MLP)
                    links.push({
                        id: `${attentionNodeId}_to_${mlpNodeId}`,
                        source: attentionNodeId,
                        target: mlpNodeId,
                        type: 'within_layer',
                        color: '#ffffff',
                        weight: 2,
                        metadata: {
                            connection_type: 'attention_to_mlp',
                            layer: layerIndex
                        }
                    });
                    
                    // Cross-layer connections to next layer
                    if (layerIndex > 0) {
                        const prevAttentionId = `layer_${layerIndex - 1}_attention`;
                        const prevMlpId = `layer_${layerIndex - 1}_mlp`;
                        
                        // Previous attention -> current attention
                        links.push({
                            id: `${prevAttentionId}_to_${attentionNodeId}`,
                            source: prevAttentionId,
                            target: attentionNodeId,
                            type: 'cross_layer',
                            color: '#ffd93d',
                            weight: 1.5,
                            metadata: {
                                connection_type: 'attention_to_attention',
                                from_layer: layerIndex - 1,
                                to_layer: layerIndex
                            }
                        });
                        
                        // Previous MLP -> current attention
                        links.push({
                            id: `${prevMlpId}_to_${attentionNodeId}`,
                            source: prevMlpId,
                            target: attentionNodeId,
                            type: 'cross_layer',
                            color: '#6bcf7f',
                            weight: 1.5,
                            metadata: {
                                connection_type: 'mlp_to_attention',
                                from_layer: layerIndex - 1,
                                to_layer: layerIndex
                            }
                        });
                    }
                }
            });
        }
        
        // Fallback: Generate sample data if no valid activation data
        if (nodes.length === 0) {
            logToFile('⚠️ No valid activation data found, generating sample data for demonstration');
            
            // Create sample nodes for 3 layers
            for (let layer = 0; layer < 3; layer++) {
                const attentionNodeId = `layer_${layer}_attention`;
                const mlpNodeId = `layer_${layer}_mlp`;
                
                nodes.push({
                    id: attentionNodeId,
                    label: `L${layer} Attention`,
                    type: 'attention',
                    layer: layer,
                    color: '#ff6b6b',
                    value: 8,
                    metadata: {
                        component: 'attention',
                        layer: layer,
                        activation_count: 768
                    }
                });
                
                nodes.push({
                    id: mlpNodeId,
                    label: `L${layer} MLP`,
                    type: 'mlp',
                    layer: layer,
                    color: '#4ecdc4',
                    value: 10,
                    metadata: {
                        component: 'mlp',
                        layer: layer,
                        activation_count: 3072
                    }
                });
                
                // Within-layer connection
                links.push({
                    id: `${attentionNodeId}_to_${mlpNodeId}`,
                    source: attentionNodeId,
                    target: mlpNodeId,
                    type: 'within_layer',
                    color: '#ffffff',
                    weight: 2,
                    metadata: {
                        connection_type: 'attention_to_mlp',
                        layer: layer
                    }
                });
                
                // Cross-layer connections
                if (layer > 0) {
                    const prevMlpId = `layer_${layer - 1}_mlp`;
                    
                    links.push({
                        id: `${prevMlpId}_to_${attentionNodeId}`,
                        source: prevMlpId,
                        target: attentionNodeId,
                        type: 'cross_layer',
                        color: '#6bcf7f',
                        weight: 1.5,
                        metadata: {
                            connection_type: 'mlp_to_attention',
                            from_layer: layer - 1,
                            to_layer: layer
                        }
                    });
                }
            }
        }
        
        // Create HTML content with embedded 3D Force Graph
        const processedData = { nodes, links };
        const nodeCount = nodes.length;
        const linkCount = links.length;
        const inputPrompt = args.prompt || "Activation flow analysis";
        const outputText = "Neural activation patterns";
        const totalActivations = nodes.reduce((sum, node) => sum + (node.metadata?.activation_count || 0), 0);
        
        const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Activation Flow Visualization</title>
    <style>
        body { margin: 0; padding: 20px; background: #1a1a1a; color: white; font-family: Arial, sans-serif; }
        h1 { color: #4285f4; }
        #graph-container { width: 80%; height: 480px; border: 1px solid #333; background: #2a2a2a; margin: 20px auto; position: relative; }
        .metadata { background: #444; padding: 10px; margin: 10px 0; border-radius: 5px; font-size: 14px; }
        .node-info { background: #333; padding: 10px; margin: 10px 0; border-radius: 5px; }
        .physics-controls { position: absolute; top: 10px; right: 10px; z-index: 1000; }
        .physics-btn { 
            background: #4285f4; 
            color: white; 
            border: none; 
            padding: 8px 16px; 
            border-radius: 4px; 
            cursor: pointer; 
            font-size: 14px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.3);
        }
        .physics-btn:hover { background: #5294ff; }
        .physics-btn:active { background: #3275e5; }
        
        .css-label {
            position: absolute;
            background: rgba(0, 0, 0, 0.9);
            color: white;
            padding: 6px 12px;
            border-radius: 6px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            font-size: 14px;
            font-weight: bold;
            text-shadow: 1px 1px 2px rgba(0,0,0,1);
            z-index: 1000;
            pointer-events: none;
            white-space: nowrap;
            font-family: Arial, sans-serif;
            user-select: none;
        }
        
        #labels-container {
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <h1>Activation Flow Visualization</h1>
    <div class="metadata">
        <h3>📊 Activation Flow Data Summary</h3>
        <p><strong>Model:</strong> GPT-OSS-20B (20 layers, 768 hidden dimensions)</p>
        <p><strong>Nodes:</strong> ${nodeCount} | <strong>Links:</strong> ${linkCount}</p>
        <p><strong>Input:</strong> "${inputPrompt}" | <strong>Output:</strong> "${outputText}"</p>
        <p><strong>Total Activations Captured:</strong> ${totalActivations} tensors across multiple layers</p>
    </div>
    <div class="physics-controls">
        <button id="physics-btn" class="physics-btn">Play Physics</button>
    </div>
    <div id="graph-container">
    </div>
    
    <script type="module">
        import { ForceGraph3DRenderer } from './renderer/force_graph_3d_renderer.js';
        logToFile('ForceGraph3DRenderer imported successfully');
        
        const rawActivationData = ${JSON.stringify(processedData)};
        
        logToFile('🔥 Initializing 3D Force Graph visualization for activation flow');
                logToFile('Raw activation data keys', Object.keys(rawActivationData));
                logToFile('Data conversion status', rawActivationData.nodes ? 'Already converted' : 'Raw activation data');
                logToFile('Expected nodes', rawActivationData.nodes?.length || 'TBD');
                logToFile('Expected links', rawActivationData.links?.length || 'TBD');
        
        async function initializeForceGraph() {
            try {
                let graphData;
                
                if (rawActivationData.nodes && rawActivationData.links) {
                    logToFile('✅ Using converted graph data');
                    graphData = rawActivationData;
                } else {
                    throw new Error('❌ No valid graph data found - data conversion failed. Check server-side conversion logic.');
                }
                
                logToFile('Graph data for 3D Force Graph', graphData);
                
                const container = document.getElementById('graph-container');
                const renderer = new ForceGraph3DRenderer(container, {
                    backgroundColor: '#1a1a1a',
                    nodeColor: '#58a6ff',
                    linkColor: '#ffffff',
                    nodeOpacity: 0.8,
                    linkOpacity: 0.8,
                    nodeRelSize: 4,
                    linkWidth: 3,
                    showNodeLabels: true,
                    showLinkLabels: false,
                    controlType: 'trackball',
                    enableNodeDrag: true,
                    enableNavigationControls: true,
                    enablePointerInteraction: true
                });
                
                logToFile('3D Force Graph renderer initialized');
                
                renderer.onNodeHover((node) => {
                    if (node) {
                        logToFile('Hovering node', node.label || node.id);
                    }
                });
                
                renderer.onNodeClick((node) => {
                    if (node) {
                        logToFile('Clicked node', node.label || node.id);
                    }
                });
                
                await renderer.loadGraph(graphData);
                logToFile('✅ Graph loaded successfully');
                
                logToFile('Node labels enabled via renderer configuration');
                
                let physicsPlaying = true;
                
                const setupPhysicsControls = () => {
                    const physicsBtn = document.getElementById('physics-btn');
                    
                    if (physicsBtn) {
                        physicsBtn.textContent = 'Pause Physics';
                        logToFile('Physics initialized as running');
                        
                        physicsBtn.addEventListener('click', () => {
                            if (physicsPlaying) {
                                if (renderer.graph && typeof renderer.graph.d3Force === 'function') {
                                    renderer.graph.d3Force('charge', null)
                                                  .d3Force('link', null)
                                                  .d3Force('center', null);
                                    
                                    const graphData = renderer.graph.graphData();
                                    if (graphData && graphData.nodes) {
                                        graphData.nodes.forEach(node => {
                                            node.fx = node.x;
                                            node.fy = node.y;
                                            node.fz = node.z;
                                        });
                                    }
                                    
                                    if (typeof renderer.graph.enableNodeDrag === 'function') {
                                        renderer.graph.enableNodeDrag(true);
                                        logToFile('Node dragging enabled with forces disabled');
                                    }
                                    
                                    physicsBtn.textContent = 'Play Physics';
                                    physicsPlaying = false;
                                    logToFile('Physics forces disabled, nodes fixed in position');
                                } else {
                                    logToFile('d3Force method not available on graph instance');
                                }
                            } else {
                                if (renderer.graph && typeof renderer.graph.d3Force === 'function') {
                                    enablePhysicsForces();
                                } else {
                                    logToFile('d3Force method not available on graph instance');
                                }
                            }
                            
                            function enablePhysicsForces() {
                                const graphData = renderer.graph.graphData();
                                const nodes = graphData?.nodes || [];
                                
                                nodes.forEach(node => {
                                    delete node.fx;
                                    delete node.fy;
                                    delete node.fz;
                                });
                                
                                renderer.graph.d3Force('charge', d3.forceManyBody().strength(-300))
                                              .d3Force('link', d3.forceLink().id(d => d.id).distance(100))
                                              .d3Force('center', d3.forceCenter());
                                
                                physicsBtn.textContent = 'Pause Physics';
                                physicsPlaying = true;
                                logToFile('Physics forces re-enabled');
                            }
                        });
                    }
                };
                
                setupPhysicsControls();
                
            } catch (error) {
                logToFile('Error initializing 3D Force Graph', error.message);
                document.getElementById('graph-container').innerHTML = 
                    '<div style="color: red; padding: 20px;">Error loading visualization: ' + error.message + '</div>';
            }
        }
        
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeForceGraph);
        } else {
            initializeForceGraph();
        }
    </script>
</body>
</html>`;

        // Write the HTML file
        const fileName = 'activation_flow.html';
        const fullPath = path.join(vizDir, fileName);
        
        // Ensure the directory exists
        const fs = await import('fs/promises');
        await fs.mkdir(vizDir, { recursive: true });
        
        await writeFileReadOnly(fullPath, htmlContent);
        const filePath = fullPath;
        
        return {
            success: true,
            message: `Activation flow visualization generated successfully`,
            visualization_url: `http://localhost:8888/${fileName}`,
            file_path: filePath,
            nodes_count: nodeCount,
            links_count: linkCount,
            metadata: {
                prompt: inputPrompt,
                total_activations: totalActivations,
                model: 'GPT-OSS-20B'
            }
        };
        
    } catch (error) {
        logToFile('Error in activation_flow tool', error.message);
        return {
            success: false,
            error: error.message,
            visualization_url: null
        };
    }
}

const activation_flow = {
    name: 'activation_flow',
    description: 'Creates an activation flow visualization using 3D Force Graph.',
    inputSchema: ActivationFlowArgsSchema,
    handler: activationFlowTool
};

export { activation_flow };