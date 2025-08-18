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
      
      // AGENT.md: Never fake anything. Report exact parameters and error details.
      throw new Error(`Graph comparison not yet implemented. No mock data per AGENT.md guidelines. Requested graphs: ${graphIds.join(', ')}. Real implementation requires fetching and comparing actual graph data.`);
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
        throw new Error('MLX Client not available');
      }
    } catch (error) {
      console.error('Failed to fetch circuit data:', error);
      throw error;
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
        throw new Error('MLX Client not available');
      }
    } catch (error) {
      console.error('Failed to fetch activation data:', error);
      throw error;
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
        throw new Error('MLX Client not available');
      }
    } catch (error) {
      console.error('Failed to fetch attention data:', error);
      throw error;
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
        throw new Error('MLX Client not available');
      }
    } catch (error) {
      console.error('Failed to fetch activation flow data:', error);
      throw error;
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
        throw new Error('MLX Client not available');
      }
    } catch (error) {
      console.error('Failed to fetch model info:', error);
      throw error;
    }
  }


}