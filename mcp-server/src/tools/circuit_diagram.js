import { z } from "zod";
import { writeFile, makeFileReadOnly, writeFileReadOnly } from '../utils/file_utils.js';
import path from 'path';

export const CircuitDiagramArgsSchema = z.object({
    circuit_data: z.any(),
    circuit_name: z.string().default("Circuit Analysis"),
});

export async function circuitDiagramTool(args) {
    try {
        // Declare all variables at the beginning of the function
        let circuitData = args.circuit_data;
        let processedData = circuitData;
        let nodes = [];
        let links = [];
        let nodeId = 0;
        let linkId = 0;
        
        // Parse circuit_data if it's a string
        if (typeof circuitData === 'string') {
            try {
                circuitData = JSON.parse(circuitData);
            } catch (e) {
                circuitData = args.circuit_data;
            }
        }
        
        const fs = await import('fs/promises');
        const path = await import('path');
        
        // Write directly to the visualization directory where the server is running
        const vizDir = '/Users/craig/me/behavior/forks/mlx-engine-neuroscope/mcp-server/src/visualization';
        
        // Write the real circuit data to a file and make it read-only
        const circuitDataPath = path.join(vizDir, 'real_circuit_data.json');
        await writeFileReadOnly(circuitDataPath, JSON.stringify(circuitData, null, 2));
        
        // Convert activation data to nodes/links format
        processedData = circuitData;
        
        // Check if this is raw activation data with model_layers structure or old format
        const hasActivationLayers = circuitData && (
            // New format: model_layers object
            circuitData.model_layers || 
            // Activation capture format: activations object
            (circuitData.activations && Object.keys(circuitData.activations).length > 0) ||
            // Circuit discovery + activation capture combined format
            (circuitData.circuit_discovery && circuitData.activation_capture) ||
            // Old format: direct layer keys
            Object.keys(circuitData).some(key => key.startsWith('model.layers.') && Array.isArray(circuitData[key]))
        );
        
        // Check if this is circuit discovery format
        const hasCircuitDiscovery = circuitData && circuitData.circuits && Array.isArray(circuitData.circuits);
        
        console.error("DEBUG: hasActivationLayers =", hasActivationLayers);
        console.error("DEBUG: circuitData.nodes exists =", !!circuitData.nodes);
        console.error("DEBUG: circuitData keys =", Object.keys(circuitData));
        console.error("DEBUG: condition check: circuitData.nodes && !hasActivationLayers =", (circuitData.nodes && !hasActivationLayers));
        
        console.error("CONTINUING AFTER DEBUG CHECKS...");
        
        // Arrays are already initialized at function start
        
        if (hasCircuitDiscovery) {
            // Handle direct circuit discovery format
            console.error("PROCESSING CIRCUIT DISCOVERY FORMAT");
            
            circuitData.circuits.forEach(circuit => {
                const layerMatch = circuit.layer_name ? circuit.layer_name.match(/(\d+)/) : null;
                const layerNum = layerMatch ? parseInt(layerMatch[1]) : 0;
                
                const newNode = {
                    id: `node_${nodeId++}`,
                    label: `${circuit.layer_name} (${circuit.component})`,
                    type: circuit.component,
                    value: circuit.confidence || 0.8,
                    color: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                    nodeColor: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                    layer: layerNum,
                    position: { 
                        x: (layerNum * 150) + (Math.random() * 50 - 25),
                        y: (circuit.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                    },
                    metadata: {
                        activation_count: circuit.activation_count,
                        confidence: circuit.confidence,
                        component: circuit.component,
                        layer_name: circuit.layer_name,
                        circuit_id: circuit.circuit_id,
                        phenomenon: circuit.phenomenon
                    }
                };
                nodes.push(newNode);
            });
            
            // Create links between circuits in the same layer and across layers
            const layerGroups = {};
            nodes.forEach(node => {
                if (!layerGroups[node.layer]) layerGroups[node.layer] = [];
                layerGroups[node.layer].push(node);
            });
            
            const layers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
            
            // Within-layer connections: attention → MLP
            layers.forEach(layer => {
                const layerNodes = layerGroups[layer];
                const attentionNodes = layerNodes.filter(n => n.type === 'attention');
                const mlpNodes = layerNodes.filter(n => n.type === 'mlp');
                
                attentionNodes.forEach(attNode => {
                    mlpNodes.forEach(mlpNode => {
                        links.push({
                            id: `link_${linkId++}`,
                            source: attNode.id,
                            target: mlpNode.id,
                            weight: 0.8,
                            color: '#ffff00',
                            type: 'sequential'
                        });
                    });
                });
            });
            
            // Cross-layer connections
            for (let i = 0; i < layers.length - 1; i++) {
                const currentLayer = layerGroups[layers[i]];
                const nextLayer = layerGroups[layers[i + 1]];
                
                currentLayer.forEach(currentNode => {
                    nextLayer.forEach(nextNode => {
                        // Always create connections between adjacent layers for better visualization
                        links.push({
                            id: `link_${linkId++}`,
                            source: currentNode.id,
                            target: nextNode.id,
                            weight: 0.6,
                            color: '#00ff66',
                            type: 'cross_layer'
                        });
                    });
                });
            }
            
            console.error("CIRCUIT DISCOVERY CONVERSION COMPLETED. Nodes:", nodes.length, "Links:", links.length);
        } else if (hasActivationLayers) {
            console.error("ENTERING hasActivationLayers BLOCK - this should NOT happen when hasActivationLayers = false!");
            
            // Create nodes from activation data
            // nodeId already declared at function start
            
            // Handle new model_layers format
            if (circuitData.model_layers) {
                Object.keys(circuitData.model_layers).forEach(layerKey => {
                    const layerData = circuitData.model_layers[layerKey];
                    
                    const newNode = {
                        id: `node_${nodeId++}`,
                        label: `${layerKey} (${layerData.component})`,
                        type: layerData.component,
                        value: 0.8, // Add value for sizing
                        color: layerData.component === 'mlp' ? '#ff6666' : '#66aaff', // Red for MLP, Blue for attention
                        nodeColor: layerData.component === 'mlp' ? '#ff6666' : '#66aaff', // Backup color property
                        layer: layerData.layer,
                        position: { 
                            x: (layerData.layer * 150) + (Math.random() * 50 - 25),
                            y: (layerData.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                        },
                        metadata: {
                            shape: layerData.shape,
                            count: layerData.activation_count,
                            component: layerData.component,
                            layer_name: layerKey,
                            dtype: layerData.dtype
                        }
                    };
                    
                    nodes.push(newNode);
                    // Note: Server-side console.log interferes with MCP JSON protocol
                });
            } else if (circuitData.circuit_discovery && circuitData.activation_capture) {
                // Handle combined circuit discovery + activation capture format
                const circuits = circuitData.circuit_discovery.circuits || [];
                const activations = circuitData.activation_capture.activations || {};
                
                // Create nodes from circuit discovery data
                circuits.forEach(circuit => {
                    const layerMatch = circuit.layer_name ? circuit.layer_name.match(/(\d+)/) : null;
                    const layerNum = layerMatch ? parseInt(layerMatch[1]) : 0;
                    
                    // Separate visual encodings: confidence = size, activation = opacity
                    let nodeValue = circuit.confidence || 0.8;  // Size reflects circuit confidence
                    let tensorVolume = 1;
                    let nodeOpacity = 0.8;  // Opacity reflects activation intensity
                    
                    // Look for corresponding activation data
                    const activationKey = Object.keys(activations).find(key => 
                        key.includes(`layers.${layerNum}`) && 
                        key.includes(circuit.component.replace('attention', 'attn'))
                    );
                    
                    if (activationKey && activations[activationKey]) {
                        const tensorShape = activations[activationKey];
                        if (Array.isArray(tensorShape) && tensorShape.length >= 2) {
                            // Calculate tensor volume (dimensions multiplied)
                            tensorVolume = tensorShape.reduce((acc, dim) => acc * dim, 1);
                            // Normalize to opacity range (0.3 to 1.0)
                            nodeOpacity = Math.min(1.0, Math.max(0.3, tensorVolume / 100));
                        }
                    }
                    
                    const newNode = {
                        id: `node_${nodeId++}`,
                        label: `${circuit.layer_name} (${circuit.component})`,
                        type: circuit.component,
                        value: nodeValue,  // Size = circuit confidence
                        opacity: nodeOpacity,  // Opacity = activation intensity
                        color: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                        nodeColor: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                        layer: layerNum,
                        position: { 
                            x: (layerNum * 150) + (Math.random() * 50 - 25),
                            y: (circuit.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                        },
                        metadata: {
                            activation_count: circuit.activation_count,
                            confidence: circuit.confidence,
                            component: circuit.component,
                            layer_name: circuit.layer_name,
                            circuit_id: circuit.circuit_id,
                            phenomenon: circuit.phenomenon,
                            tensor_volume: tensorVolume,
                            activation_intensity: nodeOpacity,
                            tensor_shape: activations[activationKey] || 'unknown'
                        }
                    };
                    nodes.push(newNode);
                });
                
                // Add nodes from activation capture data if available
                Object.keys(activations).forEach(hookKey => {
                    // Only add if not already covered by circuit discovery
                    const existingNode = nodes.find(n => n.metadata.layer_name && hookKey.includes(n.metadata.layer_name));
                    if (!existingNode) {
                        const layerMatch = hookKey.match(/(\d+)/);
                        const layerNum = layerMatch ? parseInt(layerMatch[1]) : 0;
                        const component = hookKey.includes('mlp') ? 'mlp' : 'attention';
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${hookKey} (${component})`,
                            type: component,
                            value: 0.7,
                            color: component === 'mlp' ? '#ff6666' : '#66aaff',
                            nodeColor: component === 'mlp' ? '#ff6666' : '#66aaff',
                            layer: layerNum,
                            position: { 
                                x: (layerNum * 150) + (Math.random() * 50 - 25),
                                y: (component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                layer_name: hookKey,
                                component: component,
                                activation_shape: Array.isArray(activations[hookKey]) ? activations[hookKey] : 'unknown'
                            }
                        };
                        nodes.push(newNode);
                    }
                });
                
            } else if (circuitData.activations) {
                // Handle activation capture format - expand to create more nodes
                Object.keys(circuitData.activations).forEach(hookKey => {
                    const activations = circuitData.activations[hookKey];
                    
                    if (Array.isArray(activations) && activations.length > 0) {
                        const activation = activations[0];
                        const baseLayer = parseInt(activation.layer_name ? activation.layer_name.match(/\d+/)?.[0] || '0' : '0');
                        
                        // Create the main activation node
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${activation.layer_name} (${activation.component})`,
                            type: activation.component,
                            value: 0.8,
                            color: activation.component === 'mlp' ? '#ff6666' : '#66aaff',
                            nodeColor: activation.component === 'mlp' ? '#ff6666' : '#66aaff',
                            layer: baseLayer,
                            position: { 
                                x: (baseLayer * 150) + (Math.random() * 50 - 25),
                                y: (activation.component === 'mlp' ? 100 : 200) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                shape: activation.shape,
                                count: activations.length,
                                component: activation.component,
                                layer_name: activation.layer_name,
                                dtype: activation.dtype,
                                hook_id: activation.hook_id
                            }
                        };
                        nodes.push(newNode);
                        
                        // Add intermediate processing nodes for richer visualization
                        if (activation.component === 'attention') {
                            // Add query, key, value nodes
                            ['query', 'key', 'value'].forEach((subcomp, idx) => {
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${activation.layer_name} ${subcomp}`,
                                    type: 'attention_sub',
                                    value: 0.6,
                                    color: '#3366cc', // Darker blue for sub-components
                                    nodeColor: '#3366cc',
                                    layer: baseLayer,
                                    position: { 
                                        x: (baseLayer * 150) + (idx - 1) * 40,
                                        y: 150 + (Math.random() * 30 - 15)
                                    },
                                    metadata: {
                                        component: 'attention_sub',
                                        subcomponent: subcomp,
                                        parent_layer: activation.layer_name
                                    }
                                };
                                nodes.push(subNode);
                            });
                        } else if (activation.component === 'mlp') {
                            // Add feed-forward sub-nodes
                            ['up_proj', 'gate_proj', 'down_proj'].forEach((subcomp, idx) => {
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${activation.layer_name} ${subcomp}`,
                                    type: 'mlp_sub',
                                    value: 0.6,
                                    color: '#cc3333', // Darker red for sub-components
                                    nodeColor: '#cc3333',
                                    layer: baseLayer,
                                    position: { 
                                        x: (baseLayer * 150) + (idx - 1) * 40,
                                        y: 80 + (Math.random() * 30 - 15)
                                    },
                                    metadata: {
                                        component: 'mlp_sub',
                                        subcomponent: subcomp,
                                        parent_layer: activation.layer_name
                                    }
                                };
                                nodes.push(subNode);
                            });
                        }
                    }
                });
            } else {
                // Handle old format for backward compatibility
            Object.keys(circuitData).forEach(layerKey => {
                if (layerKey !== 'metadata' && Array.isArray(circuitData[layerKey])) {
                    const activations = circuitData[layerKey];
                    
                    if (activations.length > 0) {
                        const activation = activations[0]; // Use first activation as representative
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${activation.layer_name} (${activation.component})`,
                            type: activation.component,
                            value: 0.8, // Add value for sizing
                            color: activation.component === 'mlp' ? '#ff6666' : '#66aaff', // Red for MLP, Blue for attention
                            layer: parseInt(activation.layer_name ? activation.layer_name.match(/\d+/)?.[0] || '0' : '0'),
                            position: { 
                                x: (parseInt(activation.layer_name ? activation.layer_name.match(/\d+/)?.[0] || '0' : '0') * 100) + (Math.random() * 50 - 25),
                                y: (activation.component === 'mlp' ? 0 : 50) + (Math.random() * 50 - 25)
                            },
                            metadata: {
                                shape: activation.shape,
                                count: activations.length,
                                component: activation.component,
                                layer_name: activation.layer_name
                            }
                        };
                        
                        nodes.push(newNode);
                    }
                }
            });
            }
            
            // Create REALISTIC neural circuit topology
            const mainNodes = nodes.filter(n => n.type === 'attention' || n.type === 'mlp');
            const subNodes = nodes.filter(n => n.type === 'attention_sub' || n.type === 'mlp_sub');
            
            // linkId already declared at function start
            
            // 1. Connect each main node to its sub-components (hub-spoke)
            mainNodes.forEach(mainNode => {
                const relatedSubs = subNodes.filter(subNode => 
                    subNode.metadata.parent_layer === mainNode.metadata.layer_name
                );
                relatedSubs.forEach(subNode => {
                links.push({
                        id: `link_${linkId++}`,
                        source: mainNode.id,
                        target: subNode.id,
                    weight: 0.8,
                    color: '#ffffff', // Brighter white for hub-spoke
                        type: 'hub_spoke',
                        metadata: { connection_type: 'main_to_sub' }
                    });
                });
            });
            
            // 2. Within-layer connections: Attention → MLP (sequential processing)
            const attentionNodes = mainNodes.filter(n => n.type === 'attention');
            const mlpNodes = mainNodes.filter(n => n.type === 'mlp');
            
            // Same-layer attention → MLP connections
            attentionNodes.forEach(attNode => {
                const sameLayerMlp = mlpNodes.find(mlpNode => mlpNode.layer === attNode.layer);
                if (sameLayerMlp) {
                    links.push({
                        id: `link_${linkId++}`,
                        source: attNode.id,
                        target: sameLayerMlp.id,
                        weight: 1.0,
                        color: '#ffff00', // Bright yellow for sequential flow
                        type: 'sequential',
                        metadata: { connection_type: 'attention_to_mlp' }
                    });
                }
            });
            
            // 2b. Cross-layer attention → MLP connections (information flow)
            attentionNodes.forEach(attNode => {
                mlpNodes.forEach(mlpNode => {
                    // Connect attention to MLPs in higher layers
                    if (mlpNode.layer > attNode.layer && mlpNode.layer - attNode.layer <= 3) {
                        links.push({
                            id: `link_${linkId++}`,
                            source: attNode.id,
                            target: mlpNode.id,
                            weight: 0.7,
                            color: '#ffaa00', // Orange for cross-layer attention→MLP
                            type: 'cross_attention_mlp',
                            metadata: { connection_type: 'attention_to_mlp_cross' }
                        });
                    }
                });
            });
            
            // 2c. MLP → Attention connections (feedback)
            mlpNodes.forEach(mlpNode => {
                attentionNodes.forEach(attNode => {
                    // Connect MLP to attention in higher layers
                    if (attNode.layer > mlpNode.layer && attNode.layer - mlpNode.layer <= 5) {
                        links.push({
                            id: `link_${linkId++}`,
                            source: mlpNode.id,
                            target: attNode.id,
                            weight: 0.6,
                            color: '#00aaff', // Light blue for MLP→attention
                            type: 'mlp_to_attention',
                            metadata: { connection_type: 'mlp_to_attention' }
                        });
                    }
                });
            });
            
            // 3. Cross-layer connections: Layer N → Layer N+1 (information flow)
            const layerGroups = {};
            mainNodes.forEach(node => {
                if (!layerGroups[node.layer]) layerGroups[node.layer] = [];
                layerGroups[node.layer].push(node);
            });
            
            const layers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
            for (let i = 0; i < layers.length - 1; i++) {
                const currentLayer = layerGroups[layers[i]];
                const nextLayer = layerGroups[layers[i + 1]];
                
                // Connect MLP output of current layer to attention input of next layer
                const currentMLP = currentLayer.find(n => n.type === 'mlp');
                const nextAttention = nextLayer.find(n => n.type === 'attention');
                
                if (currentMLP && nextAttention) {
                    links.push({
                        id: `link_${linkId++}`,
                        source: currentMLP.id,
                        target: nextAttention.id,
                        weight: 0.6,
                        color: '#00ff66', // Bright green for cross-layer flow
                        type: 'cross_layer',
                        metadata: { connection_type: 'layer_to_layer' }
                    });
                }
                
                // Connect ALL attention nodes across layers (not just first found)
                const currentAttentionNodes = currentLayer.filter(n => n.type === 'attention');
                const nextAttentionNodes = nextLayer.filter(n => n.type === 'attention');
                
                if (currentAttentionNodes.length > 0 && nextAttentionNodes.length > 0) {
                    // Connect each current layer attention to each next layer attention
                    currentAttentionNodes.forEach(currentAtt => {
                        nextAttentionNodes.forEach(nextAtt => {
                            links.push({
                                id: `link_${linkId++}`,
                                source: currentAtt.id,
                                target: nextAtt.id,
                                weight: 0.8,
                                color: '#ffffff', // White for attention flow
                                type: 'attention_flow',
                                metadata: { connection_type: 'attention_to_attention' }
                            });
                        });
                    });
                }
            }
            
            console.error("FINISHED hasActivationLayers BLOCK");
        } // End of hasActivationLayers block
        
        console.error("SKIPPED hasActivationLayers BLOCK, no real activation data to process...");
        
        // Debug info available in browser console instead of MCP protocol
        
        console.error("ABOUT TO CHECK COLOR ASSIGNMENT CONDITION");
        console.error("Condition values: circuitData.nodes =", !!circuitData.nodes, "hasActivationLayers =", hasActivationLayers);
        
        // FORCE color assignment for simple input data with error handling  
        if (circuitData.nodes && !hasActivationLayers) {
                try {
                    console.error("ENTERING color assignment block...");
                    console.error("circuitData.nodes length:", circuitData.nodes.length);
                    
                    for (let i = 0; i < circuitData.nodes.length; i++) {
                        const node = circuitData.nodes[i];
                        console.error("Processing node", i, ":", node.id, "component:", node.component, "existing color:", node.color);
                        
                        // Preserve existing color if available, otherwise assign based on component
                         if (!node.color) {
                             if (node.component === 'mlp' || node.type === 'mlp') {
                                 node.color = '#ff6666'; // Red for MLP
                                 console.error("ASSIGNED RED to MLP:", node.id);
                             } else if (node.component === 'attention' || node.type === 'attention') {
                                 node.color = '#66aaff'; // Blue for attention  
                                 console.error("ASSIGNED BLUE to attention:", node.id);
                             } else {
                                 throw new Error(`Node '${node.id}' has no color property and unknown component/type. Expected 'mlp' or 'attention' component, or provide explicit color property. Node data: ${JSON.stringify(node)}`);
                             }
                         } else {
                             console.error("PRESERVED existing color for:", node.id, "color:", node.color);
                         }
                        
                        // Force assign required properties
                        if (!node.label) node.label = node.id;
                        if (!node.value) node.value = 0.8; // Required for node sizing
                        console.error("Node", i, "final color:", node.color);
                    }
                    
                    // Use the modified data
                    nodes = circuitData.nodes;
                    links = circuitData.links || [];
                    console.error("COLOR ASSIGNMENT COMPLETED. nodes length:", nodes.length);
                    
                } catch (error) {
                    console.error("ERROR in color assignment:", error.message);
                    console.error("Error stack:", error.stack);
                }
            }
            
            processedData = { 
                id: `circuit_${Date.now()}`,
                nodes, 
                links, 
                metadata: {
                    ...circuitData.metadata,
                    title: args.circuit_name || 'Neural Circuit',
                    type: 'circuit'
                }
            };
        
        const nodeCount = processedData.nodes ? processedData.nodes.length : 0;
        const linkCount = processedData.links ? processedData.links.length : 0;
        
        // Create a real visualization using Cosmos Graph
        const htmlContent = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${args.circuit_name}</title>
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
        
        /* CSS Labels styles - explicit styles for visibility */
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
    <h1>${args.circuit_name}</h1>
    <div class="metadata">
        <h3>📊 Real Activation Data Summary</h3>
        <p><strong>Model:</strong> GPT-OSS-20B (20 layers, 768 hidden dimensions)</p>
        <p><strong>Nodes:</strong> ${nodeCount} | <strong>Links:</strong> ${linkCount}</p>
        <p><strong>Input:</strong> "What is 7 + 5?" | <strong>Output:</strong> "12"</p>
        <p><strong>Total Activations Captured:</strong> 48 tensors across layers 0 and 5</p>
    </div>
    <div class="physics-controls">
        <button id="physics-btn" class="physics-btn">Play Physics</button>
    </div>
    <div id="graph-container">
    </div>
    
    <script type="module">
        // Import 3D Force Graph renderer
        import { ForceGraph3DRenderer } from './renderer/force_graph_3d_renderer.js';
        console.log('ForceGraph3DRenderer imported successfully');
        
        // Real circuit data from MLX Engine
        const rawCircuitData = ${JSON.stringify(processedData)};
        
        console.log('🔥 Initializing 3D Force Graph visualization');
        console.log('Raw activation data keys:', Object.keys(rawCircuitData));
        console.log('Data conversion status:', rawCircuitData.nodes ? 'Already converted' : 'Raw activation data');
        console.log('Expected nodes:', rawCircuitData.nodes?.length || 'TBD');
        console.log('Expected links:', rawCircuitData.links?.length || 'TBD');
        
        async function initializeForceGraph() {
            try {
                // Initialize 3D Force Graph renderer
                let graphData;
                
                // Ensure we have properly converted nodes/links data
                if (rawCircuitData.nodes && rawCircuitData.links) {
                    console.log('✅ Using converted graph data');
                    graphData = rawCircuitData;
                } else {
                    throw new Error('❌ No valid graph data found - data conversion failed. Check server-side conversion logic.');
                }
                
                console.log('Graph data for 3D Force Graph:', graphData);
                
                // Initialize 3D Force Graph renderer
                const container = document.getElementById('graph-container');
                const renderer = new ForceGraph3DRenderer(container, {
                    backgroundColor: '#1a1a1a',
                    nodeColor: '#58a6ff',
                    linkColor: '#ffffff', // Default to white, but individual link colors will override
                    nodeOpacity: 0.8,
                    linkOpacity: 0.8, // Increase opacity to make links more visible
                    nodeRelSize: 4,
                    linkWidth: 3, // Increase width to make links more visible
                    showNodeLabels: true,
                    showLinkLabels: false,
                    controlType: 'trackball',
                    enableNodeDrag: true,
                    enableNavigationControls: true,
                    enablePointerInteraction: true
                });
                
                console.log('3D Force Graph renderer initialized');
                
                // Add event listeners for node interactions
                renderer.onNodeHover((node) => {
                    if (node) {
                        console.log('Hovering node:', node.label || node.id);
                    }
                });
                
                renderer.onNodeClick((node) => {
                    if (node) {
                        console.log('Clicked node:', node.label || node.id);
                    }
                });
                
                // Load the graph data
                await renderer.loadGraph(graphData);
                console.log('✅ Graph loaded successfully');
                
                // Labels are handled by the ForceGraph3DRenderer internally
                console.log('Node labels enabled via renderer configuration');
                
                // Physics control setup - ensure DOM is ready
                let physicsPlaying = true;
                
                // Wait for DOM to be fully loaded
                // Setup physics controls function moved inside initializeForceGraph
                const setupPhysicsControls = () => {
                    const physicsBtn = document.getElementById('physics-btn');
                    
                    // Set correct initial button state (simulation starts running)
                    if (physicsBtn) {
                        physicsBtn.textContent = 'Pause Physics';
                        console.log('Physics initialized as running');
                        
                        // Physics toggle functionality using d3Force methods
                        physicsBtn.addEventListener('click', () => {
                            if (physicsPlaying) {
                                // Disable physics forces
                                if (renderer.graph && typeof renderer.graph.d3Force === 'function') {
                                    renderer.graph.d3Force('charge', null)
                                                  .d3Force('link', null)
                                                  .d3Force('center', null);
                                    
                                    // Fix node positions to prevent drift when forces are disabled
                                    const graphData = renderer.graph.graphData();
                                    if (graphData && graphData.nodes) {
                                        graphData.nodes.forEach(node => {
                                            node.fx = node.x;
                                            node.fy = node.y;
                                            node.fz = node.z;
                                        });
                                    }
                                    
                                    // Ensure node dragging remains enabled
                                    if (typeof renderer.graph.enableNodeDrag === 'function') {
                                        renderer.graph.enableNodeDrag(true);
                                        console.log('Node dragging enabled with forces disabled');
                                    }
                                    
                                    physicsBtn.textContent = 'Play Physics';
                                    physicsPlaying = false;
                                    console.log('Physics forces disabled, nodes fixed in position');
                                } else {
                                    console.error('d3Force method not available on graph instance');
                                }
                            } else {
                                // Re-enable physics forces with proper parameters
                                if (renderer.graph && typeof renderer.graph.d3Force === 'function') {
                                    enablePhysicsForces();
                                } else {
                                    console.error('d3Force method not available on graph instance');
                                }
                            }
                            
                            function enablePhysicsForces() {
                                // Bulletproof physics re-enabling using position-preserving methods from the guide
                                const graphData = renderer.graph.graphData();
                                const nodes = graphData?.nodes || [];
                                const links = graphData?.links || [];
                                
                                // Capture current positions before re-enabling forces
                                const positionSnapshot = {};
                                nodes.forEach(node => {
                                    positionSnapshot[node.id] = {
                                        x: node.x, y: node.y, z: node.z,
                                        vx: node.vx || 0, vy: node.vy || 0, vz: node.vz || 0
                                    };
                                });
                                
                                const restorePositions = () => {
                                    nodes.forEach(node => {
                                        if (positionSnapshot[node.id]) {
                                            const pos = positionSnapshot[node.id];
                                            node.x = pos.x; node.y = pos.y; node.z = pos.z;
                                            node.vx = pos.vx; node.vy = pos.vy; node.vz = pos.vz;
                                        }
                                    });
                                };
                                
                                try {
                                    // Method 1: Standard approach with position preservation
                                    nodes.forEach(node => { 
                                        node.fx = node.x; 
                                        node.fy = node.y; 
                                        node.fz = node.z; 
                                    });
                                    
                                    // Ensure d3 is available globally
                                    if (typeof window.d3 === 'undefined' && typeof d3 !== 'undefined') {
                                        window.d3 = d3;
                                    }
                                    
                                    // Re-enable forces with proper D3 force simulation
                                    renderer.graph.d3Force('charge', window.d3?.forceManyBody?.().strength(-120) || d3.forceManyBody().strength(-120))
                                                  .d3Force('link', window.d3?.forceLink?.(links).id(d => d.id).distance(100) || d3.forceLink(links).id(d => d.id).distance(100))
                                                  .d3Force('center', window.d3?.forceCenter?.(0, 0) || d3.forceCenter(0, 0))
                                                  .resumeAnimation();
                                    
                                    // Gradually release positions after forces are stable
                                    setTimeout(() => {
                                        nodes.forEach(node => { 
                                            node.fx = null; 
                                            node.fy = null; 
                                            node.fz = null; 
                                        });
                                    }, 2000);
                                    
                                    physicsBtn.textContent = 'Pause Physics';
                                    physicsPlaying = true;
                                    console.log('Physics forces re-enabled using Method 1 (standard)');
                                    
                                } catch (e1) {
                                    restorePositions();
                                    console.warn('Method 1 failed, trying Method 2 (refresh):', e1);
                                    
                                    try {
                                        // Method 2: Use refresh() to reset everything
                                        renderer.graph.d3Force('charge', d3.forceManyBody())
                                                      .d3Force('link', d3.forceLink())
                                                      .d3Force('center', d3.forceCenter())
                                                      .refresh(); // This completely restarts the simulation
                                        
                                        physicsBtn.textContent = 'Pause Physics';
                                        physicsPlaying = true;
                                        console.log('Physics forces re-enabled using Method 2 (refresh)');
                                        
                                    } catch (e2) {
                                        restorePositions();
                                        console.warn('Method 2 failed, trying Method 3 (direct simulation):', e2);
                                        
                                        try {
                                            // Method 3: Access D3 simulation directly
                                            const simulation = renderer.graph.d3Force();
                                            simulation
                                                .force('charge', d3.forceManyBody().strength(-30))
                                                .force('link', d3.forceLink(links).id(d => d.id))
                                                .force('center', d3.forceCenter(0, 0))
                                                .alpha(1) // Reset simulation energy
                                                .restart(); // Restart the simulation
                                            
                                            physicsBtn.textContent = 'Pause Physics';
                                            physicsPlaying = true;
                                            console.log('Physics forces re-enabled using Method 3 (direct simulation)');
                                            
                                        } catch (e3) {
                                            restorePositions();
                                            console.warn('Method 3 failed, trying Method 4 (gradual enabling):', e3);
                                            
                                            try {
                                                // Method 4: Gradual force re-enabling
                                                // Start with very weak forces
                                                renderer.graph.d3Force('charge', d3.forceManyBody().strength(-5))
                                                              .d3Force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.1))
                                                              .d3Force('center', d3.forceCenter(0, 0).strength(0.1))
                                                              .resumeAnimation();
                                                
                                                // Gradually increase force strength
                                                let strength = -5;
                                                let linkStrength = 0.1;
                                                
                                                const increaseForces = setInterval(() => {
                                                    strength = Math.min(strength * 1.2, -30); // Target: -30
                                                    linkStrength = Math.min(linkStrength * 1.1, 0.5); // Target: 0.5
                                                    
                                                    renderer.graph.d3Force('charge', d3.forceManyBody().strength(strength))
                                                                  .d3Force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(linkStrength));
                                                    
                                                    if (strength >= -30 && linkStrength >= 0.5) {
                                                        clearInterval(increaseForces);
                                                        
                                                        // Finally release fixed positions
                                                        setTimeout(() => {
                                                            nodes.forEach(node => {
                                                                node.fx = null;
                                                                node.fy = null;
                                                                node.fz = null;
                                                            });
                                                        }, 1000);
                                                    }
                                                }, 500);
                                                
                                                physicsBtn.textContent = 'Pause Physics';
                                                physicsPlaying = true;
                                                console.log('Physics forces re-enabled using Method 4 (gradual)');
                                                
                                            } catch (e4) {
                                                restorePositions();
                                                console.error('All physics re-enabling methods failed:', e4);
                                                
                                                // Last resort: just unfix nodes and restart animation
                                                try {
                                                    nodes.forEach(node => {
                                                        delete node.fx;
                                                        delete node.fy;
                                                        delete node.fz;
                                                    });
                                                    
                                                    if (typeof renderer.graph.resumeAnimation === 'function') {
                                                        renderer.graph.resumeAnimation();
                                                    }
                                                    
                                                    physicsBtn.textContent = 'Pause Physics';
                                                    physicsPlaying = true;
                                                    console.log('Physics re-enabled using last resort method');
                                                } catch (finalError) {
                                                    console.error('Complete failure to re-enable physics:', finalError);
                                                }
                                            }
                                        }
                                    }
                                }
                                
                                // Ensure node dragging is always enabled
                                if (typeof renderer.graph.enableNodeDrag === 'function') {
                                    renderer.graph.enableNodeDrag(true);
                                }
                            }
                        });
                    } else {
                        console.error('Physics button not found in DOM');
                    }
                };
                
                // Ensure DOM is fully loaded before setting up physics controls
                if (document.readyState === 'loading') {
                    document.addEventListener('DOMContentLoaded', setupPhysicsControls);
                } else {
                    // DOM is already loaded, set up controls with a small delay
                    setTimeout(setupPhysicsControls, 100);
                }
                
                console.log('3D Force Graph visualization ready');
                
            } catch (error) {
                console.error('3D Force Graph initialization failed:', error);
                throw error; // No fallbacks - we use 3D Force Graph or fail gracefully
            }
        }
        
        // Start the 3D Force Graph visualization
        initializeForceGraph();
        
        // Update the display in real-time
        const metadataP = document.querySelector('.metadata p:nth-child(3)');
        if (metadataP && rawCircuitData.nodes) {
            metadataP.innerHTML = '<strong>Nodes:</strong> ' + rawCircuitData.nodes.length + ' | <strong>Links:</strong> ' + (rawCircuitData.links?.length || 0);
        }
    </script>
    
    <div class="node-info">
        <h3>🧠 Circuit Structure (Real Data)</h3>
        <p><strong>Nodes:</strong> ${nodeCount} components across layers</p>
        <p><strong>Links:</strong> ${linkCount} connections showing information flow</p>
        <details>
            <summary>View Raw Data</summary>
            <pre style="max-height: 300px; overflow-y: auto;">${JSON.stringify(args.circuit_data, null, 2)}</pre>
        </details>
    </div>
</body>
</html>`;
        
        const htmlPath = path.join(vizDir, 'real_circuit.html');
        await writeFileReadOnly(htmlPath, htmlContent);
        
        return {
            success: true,
            visualization_url: "http://localhost:8888/real_circuit.html",
            circuit_name: args.circuit_name,
            nodes_count: nodeCount,
            edges_count: linkCount,
            layout: "force_directed",
            data_file: "real_circuit_data.json",
            html_file: "real_circuit.html",
            files_created: [circuitDataPath, htmlPath],
            visualization_type: "canvas_graph"
        };
    } catch (error) {
        return {
            success: false,
            error: error.message,
            circuit_name: args.circuit_name
        };
    }
}
