/**
 * Visualization Tools
 * 
 * Tools that create visualizations from analysis data and integrate
 * with the Cosmos Graph visualization system.
 */

import { getLogger } from '../utils/logging.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';
import { writeFile } from 'fs/promises';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const logger = getLogger('VizTools');

/**
 * Creates a circuit diagram visualization
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Visualization result
 */
async function vizCircuitDiagram(params) {
  logger.info(`Creating circuit diagram for: ${params.circuit_name}`);
  
  try {
    // Convert circuit data to graph format
    const graphData = convertCircuitToGraph(params.circuit_data, params.circuit_name);
    
    // Save graph data for visualization
    const vizDataPath = path.join(__dirname, '../visualization/data/circuit_graph.json');
    await writeFile(vizDataPath, JSON.stringify(graphData, null, 2));
    
    // Generate visualization URL
    const vizUrl = `http://localhost:8888/index.html?data=circuit_graph.json`;
    
    return {
      success: true,
      circuit_name: params.circuit_name,
      visualization_url: vizUrl,
      graph_data: graphData,
      nodes_count: graphData.nodes.length,
      links_count: graphData.links.length,
      visualization_type: 'circuit_diagram',
    };
    
  } catch (error) {
    logger.error(`Failed to create circuit diagram: ${error.message}`);
    return {
      success: false,
      error: error.message,
      circuit_name: params.circuit_name,
    };
  }
}

/**
 * Creates an attention pattern visualization
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Visualization result
 */
async function vizAttentionPatterns(params) {
  logger.info(`Creating attention visualization for layers: ${params.layers.join(', ')}`);
  
  try {
    // Convert attention data to graph format
    const graphData = convertAttentionToGraph(params.attention_data, params.layers);
    
    // Save graph data
    const vizDataPath = path.join(__dirname, '../visualization/data/attention_graph.json');
    await writeFile(vizDataPath, JSON.stringify(graphData, null, 2));
    
    const vizUrl = `http://localhost:8888/index.html?data=attention_graph.json`;
    
    return {
      success: true,
      layers: params.layers,
      visualization_url: vizUrl,
      graph_data: graphData,
      attention_heads: graphData.metadata.attention_heads,
      pattern_types: graphData.metadata.pattern_types,
      visualization_type: 'attention_patterns',
    };
    
  } catch (error) {
    logger.error(`Failed to create attention visualization: ${error.message}`);
    return {
      success: false,
      error: error.message,
      layers: params.layers,
    };
  }
}

/**
 * Creates an activation flow visualization
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Visualization result
 */
async function vizActivationFlow(params) {
  logger.info(`Creating activation flow visualization for: "${params.prompt}"`);
  
  try {
    // Convert activation data to flow graph
    const graphData = convertActivationToFlowGraph(params.activation_data, params.prompt);
    
    // Save graph data
    const vizDataPath = path.join(__dirname, '../visualization/data/activation_flow.json');
    await writeFile(vizDataPath, JSON.stringify(graphData, null, 2));
    
    const vizUrl = `http://localhost:8888/index.html?data=activation_flow.json`;
    
    return {
      success: true,
      prompt: params.prompt,
      visualization_url: vizUrl,
      graph_data: graphData,
      layers_visualized: graphData.metadata.layers,
      flow_strength: graphData.metadata.max_flow_strength,
      visualization_type: 'activation_flow',
    };
    
  } catch (error) {
    logger.error(`Failed to create activation flow visualization: ${error.message}`);
    return {
      success: false,
      error: error.message,
      prompt: params.prompt,
    };
  }
}

/**
 * Opens the visualization interface in browser
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Result
 */
async function vizOpenBrowser(params) {
  logger.info('Opening visualization interface in browser');
  
  try {
    // Start visualization server if not running
    const vizServerPath = path.join(__dirname, '../visualization/server.js');
    
    // Check if server is already running
    const isRunning = await checkVisualizationServer();
    
    if (!isRunning) {
      // Start the server
      const serverProcess = spawn('node', [vizServerPath], {
        detached: true,
        stdio: 'ignore'
      });
      serverProcess.unref();
      
      // Wait a moment for server to start
      await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    // Open browser
    const url = params.url || 'http://localhost:8888';
    await openBrowser(url);
    
    return {
      success: true,
      url: url,
      server_status: 'running',
      message: 'Visualization interface opened in browser',
    };
    
  } catch (error) {
    logger.error(`Failed to open browser: ${error.message}`);
    return {
      success: false,
      error: error.message,
    };
  }
}

/**
 * Generates a comprehensive analysis report with visualizations
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Report result
 */
async function vizGenerateReport(params) {
  logger.info(`Generating analysis report: ${params.title}`);
  
  try {
    const report = {
      title: params.title,
      timestamp: new Date().toISOString(),
      analysis_data: params.analysis_data,
      visualizations: [],
      summary: generateAnalysisSummary(params.analysis_data),
    };
    
    // Create visualizations for different data types
    if (params.analysis_data.circuits) {
      const circuitViz = await vizCircuitDiagram({
        circuit_data: params.analysis_data.circuits,
        circuit_name: 'Analysis Circuits',
      });
      report.visualizations.push(circuitViz);
    }
    
    if (params.analysis_data.attention) {
      const attentionViz = await vizAttentionPatterns({
        attention_data: params.analysis_data.attention,
        layers: params.analysis_data.layers || [8, 10, 12],
      });
      report.visualizations.push(attentionViz);
    }
    
    if (params.analysis_data.activations) {
      const flowViz = await vizActivationFlow({
        activation_data: params.analysis_data.activations,
        prompt: params.analysis_data.prompt || 'Analysis',
      });
      report.visualizations.push(flowViz);
    }
    
    // Save report
    const reportPath = path.join(__dirname, '../visualization/data/analysis_report.json');
    await writeFile(reportPath, JSON.stringify(report, null, 2));
    
    return {
      success: true,
      title: params.title,
      report_data: report,
      visualizations_created: report.visualizations.length,
      report_url: 'http://localhost:8888/report.html',
    };
    
  } catch (error) {
    logger.error(`Failed to generate report: ${error.message}`);
    return {
      success: false,
      error: error.message,
      title: params.title,
    };
  }
}

// Helper functions for data conversion

/**
 * Converts circuit data to Cosmos Graph format
 * @param {object} circuitData - Circuit analysis data
 * @param {string} circuitName - Name of the circuit
 * @returns {object} Graph data
 */
function convertCircuitToGraph(circuitData, circuitName) {
  const nodes = [];
  const links = [];
  
  // Handle different circuit data formats
  if (Array.isArray(circuitData)) {
    // Array of circuits
    circuitData.forEach((circuit, circuitIndex) => {
      if (circuit.components) {
        circuit.components.forEach((component, compIndex) => {
          nodes.push({
            id: `${circuitIndex}_${compIndex}`,
            label: component,
            type: getComponentType(component),
            value: circuit.confidence || 0.8,
            color: getComponentColor(component),
            metadata: {
              circuit_id: circuit.id,
              circuit_name: circuit.name,
              confidence: circuit.confidence,
              layer: extractLayerFromComponent(component),
            },
          });
        });
        
        // Create links between components in the same circuit
        for (let i = 0; i < circuit.components.length - 1; i++) {
          links.push({
            id: `link_${circuitIndex}_${i}`,
            source: `${circuitIndex}_${i}`,
            target: `${circuitIndex}_${i + 1}`,
            weight: 0.8,
            type: 'circuit',
            color: '#4285f4',
          });
        }
      }
    });
  } else if (circuitData.components) {
    // Single circuit
    circuitData.components.forEach((component, index) => {
      nodes.push({
        id: `comp_${index}`,
        label: component,
        type: getComponentType(component),
        value: circuitData.confidence || 0.8,
        color: getComponentColor(component),
        metadata: {
          circuit_name: circuitName,
          confidence: circuitData.confidence,
          layer: extractLayerFromComponent(component),
        },
      });
    });
    
    // Create sequential links
    for (let i = 0; i < circuitData.components.length - 1; i++) {
      links.push({
        id: `link_${i}`,
        source: `comp_${i}`,
        target: `comp_${i + 1}`,
        weight: 0.8,
        type: 'circuit',
        color: '#4285f4',
      });
    }
  }
  
  return {
    id: `circuit_${Date.now()}`,
    nodes,
    links,
    metadata: {
      title: circuitName,
      type: 'circuit',
      created_at: new Date().toISOString(),
    },
    layout: {
      algorithm: 'force',
      parameters: { strength: 0.1, distance: 100 },
    },
    styling: {
      theme: 'dark',
      colorScheme: ['#4285f4', '#34a853', '#fbbc04', '#ea4335'],
      nodeScale: [5, 20],
      linkScale: [1, 5],
    },
  };
}

/**
 * Converts attention data to graph format
 * @param {object} attentionData - Attention analysis data
 * @param {Array} layers - Layers analyzed
 * @returns {object} Graph data
 */
function convertAttentionToGraph(attentionData, layers) {
  const nodes = [];
  const links = [];
  
  // Create nodes for tokens and attention heads
  if (attentionData.patterns) {
    attentionData.patterns.forEach((pattern, index) => {
      // Create node for attention head
      nodes.push({
        id: `head_${pattern.layer}_${pattern.head}`,
        label: `L${pattern.layer}H${pattern.head}`,
        type: 'attention_head',
        value: pattern.strength,
        color: '#4285f4',
        metadata: {
          layer: pattern.layer,
          head: pattern.head,
          pattern_type: pattern.pattern_type,
          strength: pattern.strength,
        },
      });
      
      // Create nodes for tokens involved
      if (pattern.tokens_involved) {
        pattern.tokens_involved.forEach((token, tokenIndex) => {
          const tokenId = `token_${index}_${tokenIndex}`;
          nodes.push({
            id: tokenId,
            label: token,
            type: 'token',
            value: 0.6,
            color: '#34a853',
            metadata: {
              token: token,
              position: tokenIndex,
              pattern_type: pattern.pattern_type,
            },
          });
          
          // Create link from attention head to token
          links.push({
            id: `attn_${index}_${tokenIndex}`,
            source: `head_${pattern.layer}_${pattern.head}`,
            target: tokenId,
            weight: pattern.strength,
            type: 'attention',
            color: '#cccccc',
          });
        });
      }
    });
  }
  
  return {
    id: `attention_${Date.now()}`,
    nodes,
    links,
    metadata: {
      title: 'Attention Patterns',
      type: 'attention',
      layers: layers,
      attention_heads: attentionData.patterns?.length || 0,
      pattern_types: [...new Set(attentionData.patterns?.map(p => p.pattern_type) || [])],
      created_at: new Date().toISOString(),
    },
    layout: {
      algorithm: 'hierarchical',
      parameters: { direction: 'TB' },
    },
    styling: {
      theme: 'dark',
      colorScheme: ['#4285f4', '#34a853', '#fbbc04', '#ea4335'],
      nodeScale: [3, 15],
      linkScale: [1, 3],
    },
  };
}

/**
 * Converts activation data to flow graph
 * @param {object} activationData - Activation data
 * @param {string} prompt - Input prompt
 * @returns {object} Graph data
 */
function convertActivationToFlowGraph(activationData, prompt) {
  const nodes = [];
  const links = [];
  
  // Create nodes for layers
  const layers = Object.keys(activationData).filter(key => key.startsWith('layer_'));
  layers.forEach((layerKey, index) => {
    const layerNum = parseInt(layerKey.split('_')[1]);
    nodes.push({
      id: layerKey,
      label: `Layer ${layerNum}`,
      type: 'layer',
      value: 0.8,
      color: '#4285f4',
      metadata: {
        layer: layerNum,
        activation_shape: activationData[layerKey].shape,
        activation_mean: calculateMean(activationData[layerKey].data),
      },
    });
    
    // Create links between consecutive layers
    if (index > 0) {
      const prevLayer = layers[index - 1];
      links.push({
        id: `flow_${index}`,
        source: prevLayer,
        target: layerKey,
        weight: 0.7,
        type: 'activation_flow',
        color: '#34a853',
      });
    }
  });
  
  return {
    id: `activation_flow_${Date.now()}`,
    nodes,
    links,
    metadata: {
      title: 'Activation Flow',
      type: 'activation_flow',
      prompt: prompt,
      layers: layers.length,
      max_flow_strength: 0.8,
      created_at: new Date().toISOString(),
    },
    layout: {
      algorithm: 'hierarchical',
      parameters: { direction: 'LR' },
    },
    styling: {
      theme: 'dark',
      colorScheme: ['#4285f4', '#34a853', '#fbbc04', '#ea4335'],
      nodeScale: [8, 25],
      linkScale: [2, 6],
    },
  };
}

// Utility functions

function getComponentType(component) {
  if (component.includes('attention')) return 'attention_head';
  if (component.includes('mlp')) return 'mlp';
  if (component.includes('layer')) return 'layer';
  return 'component';
}

function getComponentColor(component) {
  if (component.includes('attention')) return '#4285f4';
  if (component.includes('mlp')) return '#34a853';
  if (component.includes('layer')) return '#fbbc04';
  return '#ea4335';
}

function extractLayerFromComponent(component) {
  const match = component.match(/(\d+)/);
  return match ? parseInt(match[1]) : 0;
}

function calculateMean(data) {
  if (!Array.isArray(data) || data.length === 0) return 0;
  const flat = data.flat(Infinity);
  return flat.reduce((sum, val) => sum + val, 0) / flat.length;
}

function generateAnalysisSummary(analysisData) {
  const summary = {
    total_circuits: 0,
    total_components: 0,
    confidence_range: { min: 1, max: 0 },
    layers_analyzed: new Set(),
    analysis_types: new Set(),
  };
  
  if (analysisData.circuits) {
    summary.total_circuits = Array.isArray(analysisData.circuits) 
      ? analysisData.circuits.length 
      : 1;
      
    const circuits = Array.isArray(analysisData.circuits) 
      ? analysisData.circuits 
      : [analysisData.circuits];
      
    circuits.forEach(circuit => {
      if (circuit.components) {
        summary.total_components += circuit.components.length;
      }
      if (circuit.confidence) {
        summary.confidence_range.min = Math.min(summary.confidence_range.min, circuit.confidence);
        summary.confidence_range.max = Math.max(summary.confidence_range.max, circuit.confidence);
      }
      if (circuit.layers) {
        circuit.layers.forEach(layer => summary.layers_analyzed.add(layer));
      }
    });
  }
  
  if (analysisData.analysis_type) {
    summary.analysis_types.add(analysisData.analysis_type);
  }
  
  return {
    ...summary,
    layers_analyzed: Array.from(summary.layers_analyzed),
    analysis_types: Array.from(summary.analysis_types),
  };
}

async function checkVisualizationServer() {
  try {
    const response = await fetch('http://localhost:8888/health');
    return response.ok;
  } catch (error) {
    return false;
  }
}

async function openBrowser(url) {
  const platform = process.platform;
  let command;
  
  if (platform === 'darwin') {
    command = 'open';
  } else if (platform === 'win32') {
    command = 'start';
  } else {
    command = 'xdg-open';
  }
  
  return new Promise((resolve, reject) => {
    const process = spawn(command, [url], {
      detached: true,
      stdio: 'ignore'
    });
    
    process.on('error', reject);
    process.on('spawn', () => {
      process.unref();
      resolve();
    });
  });
}

// Tool definitions
export const vizTools = [
  {
    name: 'viz_circuit_diagram',
    description: 'Creates an interactive circuit diagram visualization',
    inputSchema: {
      type: 'object',
      properties: {
        circuit_data: {
          type: 'object',
          description: 'Circuit analysis data to visualize',
        },
        circuit_name: {
          type: 'string',
          description: 'Name of the circuit',
          default: 'Circuit Analysis',
        },
      },
      required: ['circuit_data'],
      additionalProperties: false,
    },
    handler: vizCircuitDiagram,
  },
  
  {
    name: 'viz_attention_patterns',
    description: 'Creates an attention pattern visualization',
    inputSchema: {
      type: 'object',
      properties: {
        attention_data: {
          type: 'object',
          description: 'Attention analysis data to visualize',
        },
        layers: {
          type: 'array',
          items: { type: 'integer' },
          description: 'Layers that were analyzed',
        },
      },
      required: ['attention_data', 'layers'],
      additionalProperties: false,
    },
    handler: vizAttentionPatterns,
  },
  
  {
    name: 'viz_activation_flow',
    description: 'Creates an activation flow visualization',
    inputSchema: {
      type: 'object',
      properties: {
        activation_data: {
          type: 'object',
          description: 'Activation data to visualize',
        },
        prompt: {
          type: 'string',
          description: 'Input prompt that generated the activations',
        },
      },
      required: ['activation_data', 'prompt'],
      additionalProperties: false,
    },
    handler: vizActivationFlow,
  },
  
  {
    name: 'viz_open_browser',
    description: 'Opens the visualization interface in browser',
    inputSchema: {
      type: 'object',
      properties: {
        url: {
          type: 'string',
          description: 'Specific URL to open',
          default: 'http://localhost:8888',
        },
      },
      additionalProperties: false,
    },
    handler: vizOpenBrowser,
  },
  
  {
    name: 'viz_generate_report',
    description: 'Generates a comprehensive analysis report with visualizations',
    inputSchema: {
      type: 'object',
      properties: {
        title: {
          type: 'string',
          description: 'Title of the analysis report',
        },
        analysis_data: {
          type: 'object',
          description: 'Complete analysis data to include in report',
        },
      },
      required: ['title', 'analysis_data'],
      additionalProperties: false,
    },
    handler: vizGenerateReport,
  },
];