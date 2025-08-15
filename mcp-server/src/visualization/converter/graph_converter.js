/**
 * @fileoverview Graph data converter for transforming MLX Engine data to graph format
 * Handles conversion of circuit data, attention patterns, and activation flows
 */

import { GraphTypes, NodeTypes, LinkTypes } from '../types/graph_types.js';

/**
 * Converts MLX Engine data to Cosmograph-compatible graph structures
 */
export class GraphConverter {
  /**
   * Convert circuit data to graph format
   * @param {Object} circuit - Circuit data from MLX Engine
   * @param {Object} activations - Activation data
   * @returns {import('../types/graph_types.js').GraphData}
   */
  convertCircuitToGraph(circuit, activations) {
    const nodes = [];
    const links = [];
    const nodeIdMap = new Map();

    try {
      // Process circuit components as nodes
      if (circuit.components) {
        circuit.components.forEach((component, index) => {
          const nodeId = `component_${component.id || index}`;
          nodeIdMap.set(component.id || index, nodeId);

          nodes.push({
            id: nodeId,
            label: component.name || `Component ${index}`,
            type: this._mapComponentType(component.type),
            value: this._getActivationStrength(component, activations),
            color: this._getNodeColor(component.type),
            metadata: {
              layer: component.layer,
              component: component.name,
              activation_strength: this._getActivationStrength(component, activations),
              semantic_role: component.semantic_role || 'unknown'
            }
          });
        });
      }

      // Process circuit connections as links
      if (circuit.connections) {
        circuit.connections.forEach((connection, index) => {
          const sourceId = nodeIdMap.get(connection.source);
          const targetId = nodeIdMap.get(connection.target);

          if (sourceId && targetId) {
            links.push({
              id: `connection_${index}`,
              source: sourceId,
              target: targetId,
              weight: connection.strength || 1.0,
              type: LinkTypes.CIRCUIT,
              color: this._getLinkColor(connection.strength || 1.0),
              metadata: {
                connection_type: connection.type || 'circuit',
                causal_strength: connection.strength || 1.0
              }
            });
          }
        });
      }

      return this._createGraphData({
        id: `circuit_${circuit.id || Date.now()}`,
        nodes,
        links,
        title: circuit.name || 'Circuit Visualization',
        description: circuit.description || 'Neural network circuit visualization',
        type: GraphTypes.CIRCUIT,
        model_info: circuit.model_info || {},
        analysis_info: {
          circuit_id: circuit.id,
          phenomenon: circuit.phenomenon
        }
      });
    } catch (error) {
      console.error('Error converting circuit to graph:', error);
      throw new Error(`Circuit conversion failed: ${error.message}`);
    }
  }

  /**
   * Convert attention patterns to graph format
   * @param {Object[]} attentionData - Attention pattern data
   * @returns {import('../types/graph_types.js').GraphData}
   */
  convertAttentionToGraph(attentionData) {
    const nodes = [];
    const links = [];
    const headMap = new Map();

    try {
      // Process attention heads as nodes
      attentionData.forEach((pattern, patternIndex) => {
        if (pattern.heads) {
          pattern.heads.forEach((head, headIndex) => {
            const nodeId = `head_${pattern.layer}_${headIndex}`;
            headMap.set(`${pattern.layer}_${headIndex}`, nodeId);

            nodes.push({
              id: nodeId,
              label: `L${pattern.layer}H${headIndex}`,
              type: NodeTypes.ATTENTION_HEAD,
              value: this._calculateAttentionStrength(head),
              color: this._getAttentionHeadColor(pattern.layer),
              metadata: {
                layer: pattern.layer,
                head_index: headIndex,
                attention_strength: this._calculateAttentionStrength(head),
                semantic_role: head.semantic_role || 'attention'
              }
            });
          });
        }

        // Process attention connections
        if (pattern.connections) {
          pattern.connections.forEach((connection, connIndex) => {
            const sourceId = headMap.get(`${connection.source_layer}_${connection.source_head}`);
            const targetId = headMap.get(`${connection.target_layer}_${connection.target_head}`);

            if (sourceId && targetId) {
              links.push({
                id: `attention_${patternIndex}_${connIndex}`,
                source: sourceId,
                target: targetId,
                weight: connection.weight || 1.0,
                type: LinkTypes.ATTENTION,
                color: this._getAttentionLinkColor(connection.weight || 1.0),
                metadata: {
                  connection_type: 'attention',
                  attention_weight: connection.weight || 1.0
                }
              });
            }
          });
        }
      });

      return this._createGraphData({
        id: `attention_${Date.now()}`,
        nodes,
        links,
        title: 'Attention Pattern Visualization',
        description: 'Attention head relationships and patterns',
        type: GraphTypes.ATTENTION,
        model_info: attentionData[0]?.model_info || {},
        analysis_info: {
          layer_range: this._getLayerRange(attentionData),
          phenomenon: 'attention_patterns'
        }
      });
    } catch (error) {
      console.error('Error converting attention to graph:', error);
      throw new Error(`Attention conversion failed: ${error.message}`);
    }
  }

  /**
   * Convert activation flow data to graph format
   * @param {Object} activations - Activation flow data
   * @returns {import('../types/graph_types.js').GraphData}
   */
  convertActivationsToGraph(activations) {
    const nodes = [];
    const links = [];

    try {
      // Process tokens as nodes
      if (activations.tokens) {
        activations.tokens.forEach((token, index) => {
          nodes.push({
            id: `token_${index}`,
            label: token.text || `Token ${index}`,
            type: NodeTypes.TOKEN,
            value: token.activation_strength || 1.0,
            color: this._getTokenColor(token.activation_strength || 1.0),
            metadata: {
              token_index: index,
              activation_strength: token.activation_strength || 1.0,
              semantic_role: 'token'
            }
          });
        });
      }

      // Process activation flows as links
      if (activations.flows) {
        activations.flows.forEach((flow, index) => {
          links.push({
            id: `flow_${index}`,
            source: `token_${flow.source_token}`,
            target: `token_${flow.target_token}`,
            weight: flow.strength || 1.0,
            type: LinkTypes.ACTIVATION,
            color: this._getFlowColor(flow.strength || 1.0),
            metadata: {
              connection_type: 'activation_flow',
              flow_strength: flow.strength || 1.0
            }
          });
        });
      }

      return this._createGraphData({
        id: `activation_flow_${Date.now()}`,
        nodes,
        links,
        title: 'Activation Flow Visualization',
        description: 'Token-level activation flow analysis',
        type: GraphTypes.ACTIVATION_FLOW,
        model_info: activations.model_info || {},
        analysis_info: {
          tokens: activations.tokens?.map(t => t.text) || [],
          phenomenon: 'activation_flow'
        }
      });
    } catch (error) {
      console.error('Error converting activations to graph:', error);
      throw new Error(`Activation conversion failed: ${error.message}`);
    }
  }

  /**
   * Convert model architecture to graph format
   * @param {Object} modelInfo - Model architecture information
   * @returns {import('../types/graph_types.js').GraphData}
   */
  convertModelToGraph(modelInfo) {
    const nodes = [];
    const links = [];

    try {
      // Create layer nodes
      if (modelInfo.layers) {
        modelInfo.layers.forEach((layer, index) => {
          nodes.push({
            id: `layer_${index}`,
            label: `Layer ${index}`,
            type: NodeTypes.LAYER,
            value: layer.size || 1.0,
            color: this._getLayerColor(index, modelInfo.layers.length),
            metadata: {
              layer: index,
              layer_type: layer.type || 'transformer',
              size: layer.size || 1.0,
              semantic_role: 'layer'
            }
          });

          // Create connections between consecutive layers
          if (index > 0) {
            links.push({
              id: `layer_connection_${index - 1}_${index}`,
              source: `layer_${index - 1}`,
              target: `layer_${index}`,
              weight: 1.0,
              type: LinkTypes.ACTIVATION,
              color: '#888888',
              metadata: {
                connection_type: 'layer_connection'
              }
            });
          }
        });
      }

      return this._createGraphData({
        id: `model_architecture_${Date.now()}`,
        nodes,
        links,
        title: 'Model Architecture Visualization',
        description: 'Neural network architecture overview',
        type: GraphTypes.MODEL_ARCHITECTURE,
        model_info: modelInfo,
        analysis_info: {
          phenomenon: 'model_architecture'
        }
      });
    } catch (error) {
      console.error('Error converting model to graph:', error);
      throw new Error(`Model conversion failed: ${error.message}`);
    }
  }

  /**
   * Create standardized graph data structure
   * @private
   * @param {Object} params - Graph creation parameters
   * @returns {import('../types/graph_types.js').GraphData}
   */
  _createGraphData(params) {
    return {
      id: params.id,
      nodes: params.nodes,
      links: params.links,
      metadata: {
        title: params.title,
        description: params.description,
        type: params.type,
        created_at: new Date(),
        model_info: {
          model_id: params.model_info.model_id || 'unknown',
          architecture: params.model_info.architecture || 'transformer',
          num_layers: params.model_info.num_layers || params.nodes.length
        },
        analysis_info: params.analysis_info
      },
      layout: {
        algorithm: 'force',
        parameters: {
          repulsion: 1.0,
          linkSpring: 1.0,
          linkDistance: 30
        }
      },
      styling: {
        theme: 'dark',
        colorScheme: ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a5a5f5', '#fd7e14'],
        nodeScale: [3, 15],
        linkScale: [1, 5]
      }
    };
  }

  /**
   * Map component type to node type
   * @private
   * @param {string} componentType
   * @returns {string}
   */
  _mapComponentType(componentType) {
    const typeMap = {
      'neuron': NodeTypes.NEURON,
      'attention_head': NodeTypes.ATTENTION_HEAD,
      'layer': NodeTypes.LAYER,
      'circuit': NodeTypes.CIRCUIT,
      'feature': NodeTypes.FEATURE
    };
    return typeMap[componentType] || NodeTypes.NEURON;
  }

  /**
   * Get activation strength for a component
   * @private
   * @param {Object} component
   * @param {Object} activations
   * @returns {number}
   */
  _getActivationStrength(component, activations) {
    if (!activations || !component.id) return 1.0;
    return activations[component.id] || 1.0;
  }

  /**
   * Get node color based on type (dark theme optimized)
   * @private
   * @param {string} type
   * @returns {string}
   */
  _getNodeColor(type) {
    const colorMap = {
      'neuron': '#58a6ff',      // GitHub blue
      'attention_head': '#3fb950', // GitHub green
      'layer': '#d29922',       // GitHub yellow
      'circuit': '#f85149',     // GitHub red
      'token': '#a5a5f5',       // Light purple
      'feature': '#fd7e14'      // Orange
    };
    return colorMap[type] || '#58a6ff';
  }

  /**
   * Get link color based on strength (dark theme optimized)
   * @private
   * @param {number} strength
   * @returns {string}
   */
  _getLinkColor(strength) {
    const intensity = Math.min(1.0, Math.max(0.1, strength));
    const alpha = 0.4 + (intensity * 0.6);
    return `rgba(88, 166, 255, ${alpha})`; // GitHub blue with opacity
  }

  /**
   * Get attention head color based on layer (dark theme optimized)
   * @private
   * @param {number} layer
   * @returns {string}
   */
  _getAttentionHeadColor(layer) {
    const colors = ['#58a6ff', '#3fb950', '#d29922', '#f85149', '#a5a5f5', '#fd7e14'];
    return colors[layer % colors.length];
  }

  /**
   * Get attention link color based on weight
   * @private
   * @param {number} weight
   * @returns {string}
   */
  _getAttentionLinkColor(weight) {
    const intensity = Math.min(1.0, Math.max(0.1, weight));
    const alpha = 0.2 + (intensity * 0.6);
    return `rgba(52, 168, 83, ${alpha})`;
  }

  /**
   * Calculate attention strength from head data
   * @private
   * @param {Object} head
   * @returns {number}
   */
  _calculateAttentionStrength(head) {
    if (head.attention_weights) {
      return head.attention_weights.reduce((sum, weight) => sum + weight, 0) / head.attention_weights.length;
    }
    return head.strength || 1.0;
  }

  /**
   * Get token color based on activation strength
   * @private
   * @param {number} strength
   * @returns {string}
   */
  _getTokenColor(strength) {
    const intensity = Math.min(1.0, Math.max(0.1, strength));
    const red = Math.floor(156 + (intensity * 99));
    const green = Math.floor(39 + (intensity * 40));
    const blue = Math.floor(176 + (intensity * 79));
    return `rgb(${red}, ${green}, ${blue})`;
  }

  /**
   * Get flow color based on strength
   * @private
   * @param {number} strength
   * @returns {string}
   */
  _getFlowColor(strength) {
    const intensity = Math.min(1.0, Math.max(0.1, strength));
    const alpha = 0.3 + (intensity * 0.5);
    return `rgba(156, 39, 176, ${alpha})`;
  }

  /**
   * Get layer color based on position
   * @private
   * @param {number} layerIndex
   * @param {number} totalLayers
   * @returns {string}
   */
  _getLayerColor(layerIndex, totalLayers) {
    const hue = (layerIndex / totalLayers) * 360;
    return `hsl(${hue}, 70%, 60%)`;
  }

  /**
   * Get layer range from attention data
   * @private
   * @param {Object[]} attentionData
   * @returns {[number, number]}
   */
  _getLayerRange(attentionData) {
    if (!attentionData || attentionData.length === 0) return [0, 0];
    const layers = attentionData.map(d => d.layer).filter(l => typeof l === 'number');
    return [Math.min(...layers), Math.max(...layers)];
  }
}