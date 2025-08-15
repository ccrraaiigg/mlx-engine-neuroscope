/**
 * @fileoverview Visualization generator for creating graph visualizations from MLX Engine data
 * Orchestrates data fetching, conversion, and graph generation
 */

import { GraphConverter } from '../converter/graph_converter.js';
import { GraphTypes } from '../types/graph_types.js';

/**
 * Generates visualizations by coordinating data fetching and conversion
 */
export class VisualizationGenerator {
  /**
   * @param {Object} mlxClient - MLX Engine client for data fetching
   */
  constructor(mlxClient) {
    this.mlxClient = mlxClient;
    this.converter = new GraphConverter();
  }

  /**
   * Generate circuit graph visualization
   * @param {string} circuitId - Circuit identifier
   * @returns {Promise<import('../types/graph_types.js').GraphData>}
   */
  async generateCircuitGraph(circuitId) {
    try {
      console.log(`Generating circuit graph for: ${circuitId}`);
      
      // Fetch circuit data from MLX Engine
      const circuitData = await this._fetchCircuitData(circuitId);
      const activationData = await this._fetchActivationData(circuitId);
      
      // Convert to graph format
      const graphData = this.converter.convertCircuitToGraph(circuitData, activationData);
      
      console.log(`Circuit graph generated: ${graphData.nodes.length} nodes, ${graphData.links.length} links`);
      return graphData;
    } catch (error) {
      console.error('Failed to generate circuit graph:', error);
      throw new Error(`Circuit graph generation failed: ${error.message}`);
    }
  }

  /**
   * Generate attention pattern graph visualization
   * @param {[number, number]} layerRange - Range of layers to analyze
   * @returns {Promise<import('../types/graph_types.js').GraphData>}
   */
  async generateAttentionGraph(layerRange) {
    try {
      console.log(`Generating attention graph for layers: ${layerRange[0]}-${layerRange[1]}`);
      
      // Fetch attention data from MLX Engine
      const attentionData = await this._fetchAttentionData(layerRange);
      
      // Convert to graph format
      const graphData = this.converter.convertAttentionToGraph(attentionData);
      
      console.log(`Attention graph generated: ${graphData.nodes.length} nodes, ${graphData.links.length} links`);
      return graphData;
    } catch (error) {
      console.error('Failed to generate attention graph:', error);
      throw new Error(`Attention graph generation failed: ${error.message}`);
    }
  }

  /**
   * Generate activation flow graph visualization
   * @param {string[]} tokens - Tokens to analyze
   * @returns {Promise<import('../types/graph_types.js').GraphData>}
   */
  async generateActivationFlowGraph(tokens) {
    try {
      console.log(`Generating activation flow graph for tokens: ${tokens.join(', ')}`);
      
      // Fetch activation flow data from MLX Engine
      const activationData = await this._fetchActivationFlowData(tokens);
      
      // Convert to graph format
      const graphData = this.converter.convertActivationsToGraph(activationData);
      
      console.log(`Activation flow graph generated: ${graphData.nodes.length} nodes, ${graphData.links.length} links`);
      return graphData;
    } catch (error) {
      console.error('Failed to generate activation flow graph:', error);
      throw new Error(`Activation flow graph generation failed: ${error.message}`);
    }
  }

  /**
   * Generate model architecture graph visualization
   * @param {string} modelId - Model identifier
   * @returns {Promise<import('../types/graph_types.js').GraphData>}
   */
  async generateModelArchitectureGraph(modelId) {
    try {
      console.log(`Generating model architecture graph for: ${modelId}`);
      
      // Fetch model information from MLX Engine
      const modelInfo = await this._fetchModelInfo(modelId);
      
      // Convert to graph format
      const graphData = this.converter.convertModelToGraph(modelInfo);
      
      console.log(`Model architecture graph generated: ${graphData.nodes.length} nodes, ${graphData.links.length} links`);
      return graphData;
    } catch (error) {
      console.error('Failed to generate model architecture graph:', error);
      throw new Error(`Model architecture graph generation failed: ${error.message}`);
    }
  }

  /**
   * Generate comparison graph visualization
   * @param {string[]} graphIds - Graph IDs to compare
   * @returns {Promise<import('../types/graph_types.js').GraphData>}
   */
  async generateComparisonGraph(graphIds) {
    try {
      console.log(`Generating comparison graph for: ${graphIds.join(', ')}`);
      
      // This is a placeholder for comparison graph generation
      // In a full implementation, this would fetch multiple graphs and create a comparison view
      const comparisonData = {
        id: `comparison_${Date.now()}`,
        nodes: [],
        links: [],
        metadata: {
          title: 'Graph Comparison',
          description: 'Side-by-side graph comparison',
          type: GraphTypes.COMPARISON,
          created_at: new Date(),
          model_info: {
            model_id: 'comparison',
            architecture: 'comparison',
            num_layers: 0
          },
          analysis_info: {
            compared_graphs: graphIds,
            phenomenon: 'comparison'
          }
        },
        layout: {
          algorithm: 'hierarchical',
          parameters: {
            repulsion: 0.8,
            linkSpring: 0.8,
            linkDistance: 40
          }
        },
        styling: {
          theme: 'light',
          colorScheme: ['#4285f4', '#34a853', '#fbbc04', '#ea4335'],
          nodeScale: [3, 15],
          linkScale: [1, 5]
        }
      };
      
      console.log('Comparison graph generated (placeholder)');
      return comparisonData;
    } catch (error) {
      console.error('Failed to generate comparison graph:', error);
      throw new Error(`Comparison graph generation failed: ${error.message}`);
    }
  }

  /**
   * Fetch circuit data from MLX Engine
   * @private
   * @param {string} circuitId
   * @returns {Promise<Object>}
   */
  async _fetchCircuitData(circuitId) {
    try {
      if (this.mlxClient && this.mlxClient.getCircuit) {
        return await this.mlxClient.getCircuit(circuitId);
      } else {
        // Mock data for development/testing
        return this._getMockCircuitData(circuitId);
      }
    } catch (error) {
      console.warn('Failed to fetch circuit data, using mock data:', error);
      return this._getMockCircuitData(circuitId);
    }
  }

  /**
   * Fetch activation data from MLX Engine
   * @private
   * @param {string} circuitId
   * @returns {Promise<Object>}
   */
  async _fetchActivationData(circuitId) {
    try {
      if (this.mlxClient && this.mlxClient.getActivations) {
        return await this.mlxClient.getActivations(circuitId);
      } else {
        // Mock data for development/testing
        return this._getMockActivationData(circuitId);
      }
    } catch (error) {
      console.warn('Failed to fetch activation data, using mock data:', error);
      return this._getMockActivationData(circuitId);
    }
  }

  /**
   * Fetch attention data from MLX Engine
   * @private
   * @param {[number, number]} layerRange
   * @returns {Promise<Object[]>}
   */
  async _fetchAttentionData(layerRange) {
    try {
      if (this.mlxClient && this.mlxClient.getAttentionPatterns) {
        return await this.mlxClient.getAttentionPatterns(layerRange);
      } else {
        // Mock data for development/testing
        return this._getMockAttentionData(layerRange);
      }
    } catch (error) {
      console.warn('Failed to fetch attention data, using mock data:', error);
      return this._getMockAttentionData(layerRange);
    }
  }

  /**
   * Fetch activation flow data from MLX Engine
   * @private
   * @param {string[]} tokens
   * @returns {Promise<Object>}
   */
  async _fetchActivationFlowData(tokens) {
    try {
      if (this.mlxClient && this.mlxClient.getActivationFlow) {
        return await this.mlxClient.getActivationFlow(tokens);
      } else {
        // Mock data for development/testing
        return this._getMockActivationFlowData(tokens);
      }
    } catch (error) {
      console.warn('Failed to fetch activation flow data, using mock data:', error);
      return this._getMockActivationFlowData(tokens);
    }
  }

  /**
   * Fetch model information from MLX Engine
   * @private
   * @param {string} modelId
   * @returns {Promise<Object>}
   */
  async _fetchModelInfo(modelId) {
    try {
      if (this.mlxClient && this.mlxClient.getModelInfo) {
        return await this.mlxClient.getModelInfo(modelId);
      } else {
        // Mock data for development/testing
        return this._getMockModelInfo(modelId);
      }
    } catch (error) {
      console.warn('Failed to fetch model info, using mock data:', error);
      return this._getMockModelInfo(modelId);
    }
  }

  /**
   * Generate mock circuit data for development
   * @private
   * @param {string} circuitId
   * @returns {Object}
   */
  _getMockCircuitData(circuitId) {
    return {
      id: circuitId,
      name: `Mock Circuit ${circuitId}`,
      description: 'Mock circuit data for development',
      components: [
        { id: 'comp1', name: 'Input Layer', type: 'layer', layer: 0 },
        { id: 'comp2', name: 'Hidden Layer 1', type: 'layer', layer: 1 },
        { id: 'comp3', name: 'Hidden Layer 2', type: 'layer', layer: 2 },
        { id: 'comp4', name: 'Output Layer', type: 'layer', layer: 3 }
      ],
      connections: [
        { source: 'comp1', target: 'comp2', strength: 0.8 },
        { source: 'comp2', target: 'comp3', strength: 0.6 },
        { source: 'comp3', target: 'comp4', strength: 0.9 }
      ],
      model_info: {
        model_id: 'mock-model',
        architecture: 'transformer',
        num_layers: 4
      }
    };
  }

  /**
   * Generate mock activation data for development
   * @private
   * @param {string} circuitId
   * @returns {Object}
   */
  _getMockActivationData(circuitId) {
    return {
      comp1: 0.7,
      comp2: 0.9,
      comp3: 0.5,
      comp4: 0.8
    };
  }

  /**
   * Generate mock attention data for development
   * @private
   * @param {[number, number]} layerRange
   * @returns {Object[]}
   */
  _getMockAttentionData(layerRange) {
    const data = [];
    for (let layer = layerRange[0]; layer <= layerRange[1]; layer++) {
      data.push({
        layer,
        heads: [
          { attention_weights: [0.8, 0.6, 0.4], strength: 0.6 },
          { attention_weights: [0.5, 0.9, 0.3], strength: 0.7 },
          { attention_weights: [0.7, 0.4, 0.8], strength: 0.8 }
        ],
        connections: [
          { source_layer: layer, source_head: 0, target_layer: layer, target_head: 1, weight: 0.5 },
          { source_layer: layer, source_head: 1, target_layer: layer, target_head: 2, weight: 0.7 }
        ]
      });
    }
    return data;
  }

  /**
   * Generate mock activation flow data for development
   * @private
   * @param {string[]} tokens
   * @returns {Object}
   */
  _getMockActivationFlowData(tokens) {
    return {
      tokens: tokens.map((text, index) => ({
        text,
        activation_strength: 0.5 + Math.random() * 0.5
      })),
      flows: tokens.slice(0, -1).map((_, index) => ({
        source_token: index,
        target_token: index + 1,
        strength: 0.3 + Math.random() * 0.7
      })),
      model_info: {
        model_id: 'mock-model',
        architecture: 'transformer'
      }
    };
  }

  /**
   * Generate mock model info for development
   * @private
   * @param {string} modelId
   * @returns {Object}
   */
  _getMockModelInfo(modelId) {
    return {
      model_id: modelId,
      architecture: 'transformer',
      num_layers: 12,
      layers: Array.from({ length: 12 }, (_, i) => ({
        type: 'transformer',
        size: 768 + Math.random() * 256
      }))
    };
  }
}