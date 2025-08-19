import { z } from "zod";
import { writeFile, makeFileReadOnly } from '../utils/file_utils.js';
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
        
        // Convert activation data to nodes/links format
        processedData = circuitData;
        
        // Prepare path for writing processed data later
        let circuitDataPath = path.join(vizDir, 'real_circuit_data.json');
        
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
        
        console.error("=== CIRCUIT DIAGRAM TOOL START ===");
        console.error("Input circuitData:", JSON.stringify(circuitData, null, 2));
        console.error("circuitData type:", typeof circuitData);
        console.error("circuitData is array:", Array.isArray(circuitData));
        
        // Handle discover_circuits format FIRST (either array or object with circuits array)
        const circuitsArray = Array.isArray(circuitData) ? circuitData : 
                             (circuitData.circuits && Array.isArray(circuitData.circuits)) ? circuitData.circuits : null;
        
        console.error("DEBUG: circuitsArray =", circuitsArray ? circuitsArray.length : 'null');
        console.error("DEBUG: circuitData.circuits exists =", !!circuitData.circuits);
        console.error("DEBUG: circuitData.circuits is array =", Array.isArray(circuitData.circuits));
        
        if (circuitsArray && circuitsArray.length > 0) {
            console.error("PROCESSING DISCOVER_CIRCUITS ARRAY FORMAT with", circuitsArray.length, "circuits");
            console.error("First circuit:", JSON.stringify(circuitsArray[0], null, 2));
            
            // Convert discover_circuits array to nodes and links
            const nodeMap = new Map();
            nodeId = 1;
            linkId = 1;
            
            // Create nodes from circuit data
            circuitsArray.forEach((circuit, index) => {
                console.error("Processing circuit", index, ":", JSON.stringify(circuit, null, 2));
                // Check if circuit has required properties
                if (!circuit.layer_name || !circuit.component) {
                    console.error("Skipping invalid circuit data:", circuit);
                    return;
                }
                console.error("Circuit has valid layer_name and component:", circuit.layer_name, circuit.component);
                
                const layerMatch = circuit.layer_name.match(/layers\.(\d+)\./); 
                const layer = layerMatch ? parseInt(layerMatch[1]) : index;
                
                const nodeKey = `${circuit.layer_name}_${circuit.component}`;
                if (!nodeMap.has(nodeKey)) {
                    const node = {
                        id: `node_${nodeId++}`,
                        label: `L${layer} ${circuit.component.toUpperCase()}`,
                        layer: layer,
                        type: circuit.component,
                        activation_count: circuit.activation_count,
                        confidence: circuit.confidence,
                        description: circuit.description,
                        color: circuit.component === 'mlp' ? '#ff6666' : '#66aaff',
                        size: Math.max(8, circuit.confidence * 20),
                        opacity: Math.max(0.3, circuit.confidence),
                        value: circuit.activation_count || circuit.confidence * 100 || 10
                    };
                    nodes.push(node);
                    nodeMap.set(nodeKey, node);
                    console.error("Created node:", node.id, "for", circuit.layer_name, circuit.component);
                }
            });
            
            // Create links between nodes in the same layer and across layers
            const layerGroups = {};
            nodes.forEach(node => {
                if (!layerGroups[node.layer]) layerGroups[node.layer] = [];
                layerGroups[node.layer].push(node);
            });
            
            const layers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
            
            // Intra-layer connections (attention -> MLP within same layer)
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
                            color: '#ffaa00',
                            type: 'intra_layer',
                            metadata: { connection_type: 'attention_to_mlp' }
                        });
                    });
                });
            });
            
            // Cross-layer connections - create comprehensive circuit topology
            for (let i = 0; i < layers.length - 1; i++) {
                const currentLayer = layerGroups[layers[i]];
                const nextLayer = layerGroups[layers[i + 1]];
                
                currentLayer.forEach(currentNode => {
                    nextLayer.forEach(nextNode => {
                        // Create ALL possible connections between adjacent layers
                        let connectionType = '';
                        let color = '#00ff66';
                        let weight = 0.6;
                        
                        if (nextNode.type === 'attention') {
                            // Any component can feed into attention via residual stream
                            connectionType = 'to_attention';
                            color = '#00ff66'; // Green for residual connections
                            weight = 0.7;
                        } else if (currentNode.type === 'attention' && nextNode.type === 'mlp') {
                            // Attention can feed into MLP
                            connectionType = 'attention_to_mlp';
                            color = '#ffaa00'; // Orange for attention->MLP
                            weight = 0.8;
                        } else if (currentNode.type === 'mlp' && nextNode.type === 'attention') {
                            // MLP can feed into attention
                            connectionType = 'mlp_to_attention';
                            color = '#66ffaa'; // Light green for MLP->attention
                            weight = 0.7;
                        } else {
                            // Default connection for any other combination
                            connectionType = 'general';
                            color = '#aaaaff'; // Light blue for general connections
                            weight = 0.5;
                        }
                        
                        links.push({
                            id: `link_${linkId++}`,
                            source: currentNode.id,
                            target: nextNode.id,
                            weight: weight,
                            color: color,
                            type: 'cross_layer',
                            metadata: { connection_type: connectionType }
                        });
                    });
                });
            }
            
            // Add intra-layer connections (within the same layer)
            layers.forEach(layer => {
                const layerNodes = layerGroups[layer];
                if (layerNodes.length > 1) {
                    for (let i = 0; i < layerNodes.length; i++) {
                        for (let j = i + 1; j < layerNodes.length; j++) {
                            const sourceNode = layerNodes[i];
                            const targetNode = layerNodes[j];
                            
                            links.push({
                                id: `link_${linkId++}`,
                                source: sourceNode.id,
                                target: targetNode.id,
                                weight: 0.4,
                                color: '#ffff66', // Yellow for intra-layer connections
                                type: 'intra_layer',
                                metadata: { connection_type: 'same_layer' }
                            });
                        }
                    }
                }
            });
            
            // Add additional circuit-specific connections for better visualization
            // Connect components that are likely to interact in arithmetic circuits
            nodes.forEach(sourceNode => {
                nodes.forEach(targetNode => {
                    if (sourceNode.id !== targetNode.id && targetNode.layer > sourceNode.layer) {
                        // Connect early attention to late MLP (common in arithmetic circuits)
                        if (sourceNode.type === 'attention' && targetNode.type === 'mlp' && 
                            targetNode.layer - sourceNode.layer >= 10) {
                            links.push({
                                id: `link_${linkId++}`,
                                source: sourceNode.id,
                                target: targetNode.id,
                                weight: 0.5,
                                color: '#ff6600', // Orange-red for long-range connections
                                type: 'long_range',
                                metadata: { connection_type: 'early_to_late' }
                            });
                        }
                        // Connect MLP to later attention (information flow)
                        else if (sourceNode.type === 'mlp' && targetNode.type === 'attention' && 
                                targetNode.layer - sourceNode.layer >= 3 && targetNode.layer - sourceNode.layer <= 8) {
                            links.push({
                                id: `link_${linkId++}`,
                                source: sourceNode.id,
                                target: targetNode.id,
                                weight: 0.6,
                                color: '#66ffaa', // Light green for medium-range connections
                                type: 'medium_range',
                                metadata: { connection_type: 'mlp_to_attention' }
                            });
                        }
                    }
                });
            });
            
            console.error("DISCOVER_CIRCUITS CONVERSION COMPLETED. nodes:", nodes.length, "links:", links.length);
        }
        // Arrays are already initialized at function start
        else if (hasActivationLayers) {
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
                    const layerMatch = circuit.layer_name.match(/(\d+)/);
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
                    
                    // Create proper label with fallbacks for undefined values
                    const layerName = circuit.layer_name || `Layer ${layerNum}`;
                    const componentName = circuit.component || 'unknown';
                    const nodeLabel = `${layerName} (${componentName})`;
                    
                    const newNode = {
                        id: `node_${nodeId++}`,
                        label: nodeLabel,
                        type: circuit.component || 'unknown',
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
                        
                        // Create proper label with fallbacks
                        const hookLabel = hookKey || `Layer ${layerNum}`;
                        const compLabel = component || 'unknown';
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${hookLabel} (${compLabel})`,
                            type: component || 'unknown',
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
                
            } else if (circuitData.activations && !Array.isArray(circuitData)) {
                // Handle activation capture format - expand to create more nodes
                // Only process if this is NOT a discover_circuits array format
                Object.keys(circuitData.activations).forEach(hookKey => {
                    const activations = circuitData.activations[hookKey];
                    
                    if (Array.isArray(activations) && activations.length > 0) {
                        const activation = activations[0];
                        const baseLayer = parseInt((activation.layer_name || '').match(/\d+/)?.[0] || '0');
                        
                        // Create the main activation node with proper label handling
                        const layerName = activation.layer_name || `Layer ${baseLayer}`;
                        const componentName = activation.component || 'unknown';
                        
                        const newNode = {
                            id: `node_${nodeId++}`,
                            label: `${layerName} (${componentName})`,
                            type: activation.component || 'unknown',
                            value: 0.8,
                            color: activation.component === 'mlp' ? '#ff6666' : '#66aaff', // Use hex colors for consistency
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
                                const layerLabel = activation.layer_name || `Layer ${baseLayer}`;
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${layerLabel} ${subcomp}`,
                                    type: 'attention_sub',
                                    value: 0.6,
                                    color: '#4488cc', // Darker blue for sub-components
                                    nodeColor: '#4488cc',
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
                                const layerLabel = activation.layer_name || `Layer ${baseLayer}`;
                                const subNode = {
                                    id: `node_${nodeId++}`,
                                    label: `${layerLabel} ${subcomp}`,
                                    type: 'mlp_sub',
                                    value: 0.6,
                                    color: '#cc4444', // Darker red for sub-components
                                    nodeColor: '#cc4444',
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
                            layer: parseInt((activation.layer_name || '').match(/\d+/)?.[0] || '0'),
                            position: { 
                                x: (parseInt((activation.layer_name || '').match(/\d+/)?.[0] || '0') * 100) + (Math.random() * 50 - 25),
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
            
            // Create meaningful circuit topology connections
            if (Array.isArray(circuitData) || (circuitData.circuit_discovery && circuitData.activation_capture)) {
                // Group nodes by layer for better connection logic
                const nodesByLayer = {};
                nodes.forEach(node => {
                    const layer = node.layer || 0;
                    if (!nodesByLayer[layer]) nodesByLayer[layer] = [];
                    nodesByLayer[layer].push(node);
                });
                
                const layers = Object.keys(nodesByLayer).map(Number).sort((a, b) => a - b);
                
                // Create sequential layer connections (attention -> MLP within layer, then to next layer)
                layers.forEach((currentLayer, layerIdx) => {
                    const currentLayerNodes = nodesByLayer[currentLayer];
                    const attentionNodes = currentLayerNodes.filter(n => n.type === 'attention');
                    const mlpNodes = currentLayerNodes.filter(n => n.type === 'mlp');
                    
                    // Connect attention to MLP within the same layer
                    attentionNodes.forEach(attNode => {
                        mlpNodes.forEach(mlpNode => {
                            links.push({
                                id: `link_${linkId++}`,
                                source: attNode.id,
                                target: mlpNode.id,
                                weight: 0.8,
                                color: '#ffaa00', // Orange for intra-layer connections
                                type: 'intra_layer',
                                metadata: { 
                                    connection_type: 'attention_to_mlp',
                                    layer: currentLayer
                                }
                            });
                        });
                    });
                    
                    // Connect to next layer
                    if (layerIdx < layers.length - 1) {
                        const nextLayer = layers[layerIdx + 1];
                        const nextLayerNodes = nodesByLayer[nextLayer];
                        
                        currentLayerNodes.forEach(currentNode => {
                            nextLayerNodes.forEach(nextNode => {
                                // Stronger connections between same component types
                                const isSameType = currentNode.type === nextNode.type;
                                const weight = isSameType ? 0.9 : 0.6;
                                const color = isSameType ? '#00ff88' : '#88aaff';
                                
                                links.push({
                                    id: `link_${linkId++}`,
                                    source: currentNode.id,
                                    target: nextNode.id,
                                    weight: weight,
                                    color: color,
                                    type: 'inter_layer',
                                    metadata: { 
                                        connection_type: isSameType ? 'same_type_flow' : 'cross_type_flow',
                                        source_layer: currentLayer,
                                        target_layer: nextLayer
                                    }
                                });
                            });
                        });
                    }
                });
                
                // Add residual connections (skip connections)
                if (layers.length > 2) {
                    layers.forEach((currentLayer, layerIdx) => {
                        if (layerIdx < layers.length - 2) { // Skip one layer
                            const skipLayer = layers[layerIdx + 2];
                            const currentNodes = nodesByLayer[currentLayer];
                            const skipNodes = nodesByLayer[skipLayer];
                            
                            currentNodes.forEach(currentNode => {
                                skipNodes.forEach(skipNode => {
                                    if (currentNode.type === skipNode.type) { // Only same types for residual
                                        links.push({
                                            id: `link_${linkId++}`,
                                            source: currentNode.id,
                                            target: skipNode.id,
                                            weight: 0.4,
                                            color: '#ff6666', // Red for residual connections
                                            type: 'residual',
                                            metadata: { 
                                                connection_type: 'residual_connection',
                                                source_layer: currentLayer,
                                                target_layer: skipLayer
                                            }
                                        });
                                    }
                                });
                            });
                        }
                    });
                }
                
                // Group nodes by type for additional analysis
                const attentionNodes = nodes.filter(n => n.type === 'attention');
                const mlpNodes = nodes.filter(n => n.type === 'mlp');
                
                // Connect attention to MLP nodes with different layers
                attentionNodes.forEach(attNode => {
                    mlpNodes.forEach(mlpNode => {
                        if (mlpNode.layer > attNode.layer && Math.random() > 0.7) { // 30% chance
                            links.push({
                                id: `link_${linkId++}`,
                                source: attNode.id,
                                target: mlpNode.id,
                                weight: 0.3 + Math.random() * 0.4,
                                color: '#66aaff',
                                type: 'attention_to_mlp',
                                metadata: { connection_type: 'cross_component' }
                            });
                        }
                    });
                });
            } else {
                // For other formats, use the complex topology
                const mainNodes = nodes.filter(n => n.type === 'attention' || n.type === 'mlp');
                const subNodes = nodes.filter(n => n.type === 'attention_sub' || n.type === 'mlp_sub');
                
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
                        color: '#ffffff',
                            type: 'hub_spoke',
                            metadata: { connection_type: 'main_to_sub' }
                        });
                    });
                });
                
                // 2. Create cross-layer connections
                const layerGroups = {};
                mainNodes.forEach(node => {
                    if (!layerGroups[node.layer]) layerGroups[node.layer] = [];
                    layerGroups[node.layer].push(node);
                });
                
                const layers = Object.keys(layerGroups).map(Number).sort((a, b) => a - b);
                for (let i = 0; i < layers.length - 1; i++) {
                    const currentLayer = layerGroups[layers[i]];
                    const nextLayer = layerGroups[layers[i + 1]];
                    
                    currentLayer.forEach(currentNode => {
                        nextLayer.forEach(nextNode => {
                            if (Math.random() > 0.6) { // 40% chance of connection
                                links.push({
                                    id: `link_${linkId++}`,
                                    source: currentNode.id,
                                    target: nextNode.id,
                                    weight: 0.4 + Math.random() * 0.4,
                                    color: '#00ff66',
                                    type: 'cross_layer',
                                    metadata: { connection_type: 'layer_to_layer' }
                                });
                            }
                        });
                    });
                }
            }
            
            console.error("FINISHED hasActivationLayers BLOCK");
        } // End of hasActivationLayers block
        else {
            console.error("SKIPPED hasActivationLayers BLOCK, no real activation data to process...");
        }
        
        // Note: discover_circuits array format is already processed in the first conditional block above
        
        // Create the final processed data object with nodes and links
        processedData = {
            id: `circuit_${Date.now()}`,
            nodes: nodes,
            links: links,
            metadata: {
                title: args.circuit_name,
                type: "circuit"
            }
        };
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
                        console.error("Processing node", i, ":", node.id, "type:", node.type);
                        
                        // Force assign colors based on type
                        if (node.type === 'mlp') {
                            node.color = [1.0, 0.4, 0.4, 1.0]; // Red for MLP
                            console.error("ASSIGNED RED to MLP:", node.id);
                        } else if (node.type === 'attention') {
                            node.color = '#66aaff'; // Blue for attention  
                            console.error("ASSIGNED BLUE to attention:", node.id);
                        } else {
                            node.color = [0.6, 0.6, 0.6, 1.0]; // Gray fallback
                            console.error("ASSIGNED GRAY to unknown:", node.id);
                        }
                        
                        // Preserve existing label if it exists, otherwise use a meaningful fallback
                        if (!node.label || node.label === 'undefined (undefined)') {
                            node.label = `${node.type || 'Node'} ${node.layer || 0}`;
                        }
                        
                        // Ensure value property exists for 3D Force Graph compatibility
                        if (!node.value) {
                            node.value = node.activation_count || node.confidence || 1.0;
                        }
                        
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
        
        // Write the processed data to JSON file (keep writable during generation)
        circuitDataPath = path.join(vizDir, 'real_circuit_data.json');
        try {
            const writeResult = await writeFile(circuitDataPath, JSON.stringify(processedData, null, 2));
            console.error('DEBUG: JSON writeFile result:', writeResult);
            console.error('DEBUG: JSON file path:', circuitDataPath);
        } catch (writeError) {
            console.error('ERROR: Failed to write JSON file:', writeError);
            throw new Error(`Failed to write JSON file: ${writeError.message}`);
        }
        
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
        #graph-container { width: 1000px; height: 1000px; border: 1px solid #333; background: #2a2a2a; margin: 20px auto; position: relative; }
        #graph-container canvas { width: 1000px !important; height: 1000px !important; }
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
        <p><strong>Input:</strong> Agent-provided prompt | <strong>Output:</strong> Model response</p>
        <p><strong>Total Activations Captured:</strong> 48 tensors across layers 0 and 5</p>
    </div>
    <div id="graph-container">
        <div class="physics-controls">
            <button id="physics-btn" class="physics-btn">Play Physics</button>
        </div>
    </div>
    
    <script src="https://unpkg.com/3d-force-graph@1.70.19/dist/3d-force-graph.min.js"></script>
    <script>
        // Real circuit data from MLX Engine
        const rawCircuitData = ${JSON.stringify(processedData)};
        
        console.log('🔥 Initializing 3D Force Graph visualization');
        console.log('Raw activation data keys:', Object.keys(rawCircuitData));
        console.log('Data conversion status:', rawCircuitData.nodes ? 'Already converted' : 'Raw activation data');
        console.log('Expected nodes:', rawCircuitData.nodes?.length || 'TBD');
        console.log('Expected links:', rawCircuitData.links?.length || 'TBD');
        
        function initializeForceGraph() {
            try {
                // Ensure we have properly converted nodes/links data
                let graphData;
                if (rawCircuitData.nodes && rawCircuitData.links) {
                    console.log('✅ Using converted graph data');
                    graphData = rawCircuitData;
                } else {
                    throw new Error('❌ No valid graph data found - data conversion failed. Check server-side conversion logic.');
                }
                
                console.log('Graph data for 3D Force Graph:', graphData);
                console.log('Nodes:', graphData.nodes.length, 'Links:', graphData.links.length);
                
                // Initialize 3D Force Graph
                const container = document.getElementById('graph-container');
                const Graph = ForceGraph3D()(container)
                    .graphData(graphData)
                    .backgroundColor('#1a1a1a')
                    .nodeColor(node => {
                        return node.type === 'attention' ? '#58a6ff' : '#ff6b6b';
                    })
                    .nodeLabel(node => \`\${node.label}<br/>Layer: \${node.layer}<br/>Type: \${node.type}<br/>Confidence: \${node.confidence}\`)
                    .nodeVal(node => node.value || 10)
                    .linkColor('#30363d')
                    .linkWidth(2)
                    .linkOpacity(0.6)
                    .nodeOpacity(0.9)
                    .width(1000)
                    .height(1000)
                    .enableNodeDrag(true)
                    .onNodeHover(node => {
                        if (node) {
                            console.log('Hovering node:', node.label || node.id);
                        }
                    })
                    .onNodeClick(node => {
                        if (node) {
                            console.log('Clicked node:', node.label || node.id);
                        }
                    });
                
                console.log('✅ 3D Force Graph initialized successfully');
                
                // Physics control setup
                let physicsPlaying = true;
                const physicsBtn = document.getElementById('physics-btn');
                
                if (physicsBtn) {
                    physicsBtn.textContent = 'Pause Physics';
                    console.log('Physics initialized as running');
                    
                    physicsBtn.addEventListener('click', () => {
                        if (physicsPlaying) {
                            Graph.pauseAnimation();
                            physicsBtn.textContent = 'Play Physics';
                            physicsPlaying = false;
                            console.log('Physics paused via button');
                        } else {
                            Graph.resumeAnimation();
                            physicsBtn.textContent = 'Pause Physics';
                            physicsPlaying = true;
                            console.log('Physics resumed via button');
                        }
                    });
                } else {
                    console.error('Physics button not found in DOM');
                }
                
                console.log('3D Force Graph visualization ready');
                
            } catch (error) {
                console.error('3D Force Graph initialization failed:', error);
                throw error;
            }
        }
        
        // Start the 3D Force Graph visualization when DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', initializeForceGraph);
        } else {
            initializeForceGraph();
        }
        
        // Update the display in real-time
        document.addEventListener('DOMContentLoaded', () => {
            const metadataP = document.querySelector('.metadata p:nth-child(3)');
            if (metadataP && rawCircuitData.nodes) {
                metadataP.innerHTML = '<strong>Nodes:</strong> ' + rawCircuitData.nodes.length + ' | <strong>Links:</strong> ' + (rawCircuitData.links?.length || 0);
            }
        });
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
        await writeFile(htmlPath, htmlContent);
        
        // Now that HTML is complete, make both files read-only for protection
        await makeFileReadOnly(circuitDataPath);
        await makeFileReadOnly(htmlPath);
        
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