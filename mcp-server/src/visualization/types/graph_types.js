/**
 * @fileoverview TypeScript-style interfaces and types for graph data structures
 * Used with @cosmos.gl/graph visualization library
 */

/**
 * @typedef {Object} NodeData
 * @property {string} id - Unique identifier for the node
 * @property {string} label - Display label for the node
 * @property {'neuron'|'attention_head'|'layer'|'circuit'|'token'|'feature'} type - Node type
 * @property {number} value - Size/importance value for visual representation
 * @property {string} color - Color for the node
 * @property {{x: number, y: number}} [position] - Optional fixed position
 * @property {Object} metadata - Additional metadata
 * @property {number} [metadata.layer] - Layer number if applicable
 * @property {string} [metadata.component] - Component name
 * @property {number} [metadata.activation_strength] - Activation strength
 * @property {string} [metadata.semantic_role] - Semantic role description
 */

/**
 * @typedef {Object} LinkData
 * @property {string} id - Unique identifier for the link
 * @property {string} source - Source node ID
 * @property {string} target - Target node ID
 * @property {number} weight - Connection strength/weight
 * @property {'activation'|'attention'|'circuit'|'causal'|'similarity'} type - Link type
 * @property {string} color - Color for the link
 * @property {Object} metadata - Additional metadata
 * @property {string} [metadata.connection_type] - Type of connection
 * @property {number} [metadata.attention_weight] - Attention weight if applicable
 * @property {number} [metadata.causal_strength] - Causal strength if applicable
 */

/**
 * @typedef {Object} GraphMetadata
 * @property {string} title - Graph title
 * @property {string} description - Graph description
 * @property {'circuit'|'attention'|'activation_flow'|'model_architecture'|'comparison'} type - Graph type
 * @property {Date} created_at - Creation timestamp
 * @property {Object} model_info - Model information
 * @property {string} model_info.model_id - Model identifier
 * @property {string} model_info.architecture - Model architecture
 * @property {number} model_info.num_layers - Number of layers
 * @property {Object} analysis_info - Analysis information
 * @property {string} [analysis_info.circuit_id] - Circuit identifier
 * @property {[number, number]} [analysis_info.layer_range] - Layer range
 * @property {string[]} [analysis_info.tokens] - Tokens analyzed
 * @property {string} [analysis_info.phenomenon] - Phenomenon being analyzed
 */

/**
 * @typedef {Object} LayoutConfiguration
 * @property {'force'|'hierarchical'|'circular'} algorithm - Layout algorithm
 * @property {Object} parameters - Algorithm-specific parameters
 */

/**
 * @typedef {Object} StylingConfiguration
 * @property {'light'|'dark'|'custom'} theme - Visual theme
 * @property {string[]} colorScheme - Color scheme array
 * @property {[number, number]} nodeScale - Node size scale range
 * @property {[number, number]} linkScale - Link width scale range
 */

/**
 * @typedef {Object} GraphData
 * @property {string} id - Unique graph identifier
 * @property {NodeData[]} nodes - Array of nodes
 * @property {LinkData[]} links - Array of links
 * @property {GraphMetadata} metadata - Graph metadata
 * @property {LayoutConfiguration} layout - Layout configuration
 * @property {StylingConfiguration} styling - Styling configuration
 */

/**
 * @typedef {Object} CosmosGraphConfig
 * @property {number} pointSize - Size of points (nodes)
 * @property {number} linkWidth - Width of links
 * @property {string} backgroundColor - Background color (CSS color string)
 * @property {string} pointColor - Default point color (CSS color string)
 * @property {string} linkColor - Default link color (CSS color string)
 * @property {boolean} showLabels - Whether to show labels
 * @property {number} simulationFriction - Friction coefficient for simulation
 * @property {number} simulationGravity - Gravity force for simulation
 * @property {number} simulationRepulsion - Repulsion force between nodes
 * @property {number} linkDistance - Base distance between linked nodes
 * @property {boolean} enableZoom - Whether zoom is enabled
 * @property {boolean} enablePan - Whether pan is enabled
 * @property {boolean} enableHover - Whether hover effects are enabled
 * @property {boolean} enableTooltip - Whether tooltips are enabled
 * @property {number} [minZoom] - Minimum zoom level
 * @property {number} [maxZoom] - Maximum zoom level
 */

/**
 * @typedef {Object} GraphConfig
 * Configuration for the graph visualization
 * @property {string} [backgroundColor='#0d1117'] - Background color of the graph
 * @property {string} [pointColor='#58a6ff'] - Default node color
 * @property {string} [linkColor='#30363d'] - Default link color
 * @property {number} [pointSize=4] - Default node size
 * @property {number} [linkWidth=1] - Default link width
 * @property {boolean} [showLabels=true] - Whether to show node labels
 * @property {number} [simulationFriction=0.85] - Physics simulation friction
 * @property {number} [simulationGravity=0.1] - Physics simulation gravity
 * @property {number} [simulationRepulsion=1.0] - Node repulsion force
 * @property {number} [linkDistance=50] - Default link distance
 * @property {Function} [onClick] - Click handler for nodes
 * @property {Function} [onMouseMove] - Mouse move handler for nodes
 */

/**
 * @typedef {Object} VisualizationState
 * @property {string} graph_id - Graph identifier
 * @property {number} zoom_level - Current zoom level
 * @property {{x: number, y: number}} pan_position - Current pan position
 * @property {string[]} selected_nodes - Selected node IDs
 * @property {string[]} highlighted_nodes - Highlighted node IDs
 * @property {Object[]} active_filters - Active filter configurations
 * @property {Object} layout_state - Current layout state
 */

/**
 * @typedef {Object} BrowserCapabilities
 * @property {boolean} webgl - WebGL 1.0 support
 * @property {boolean} webgl2 - WebGL 2.0 support
 * @property {number} maxTextureSize - Maximum texture size
 * @property {number} maxVertexUniforms - Maximum vertex uniforms
 * @property {'high'|'medium'|'low'} performanceLevel - Performance level
 */

/**
 * @typedef {Object} NodeUpdate
 * @property {string} id - Node ID to update
 * @property {Partial<NodeData>} data - Partial node data to update
 */

/**
 * @typedef {Object} LinkUpdate
 * @property {string} id - Link ID to update
 * @property {Partial<LinkData>} data - Partial link data to update
 */

/**
 * @typedef {Object} GraphExportData
 * @property {GraphData} graph_data - Complete graph data
 * @property {CosmosGraphConfig} cosmos_config - Cosmos Graph configuration
 * @property {Object} export_metadata - Export metadata
 * @property {Date} export_metadata.exported_at - Export timestamp
 * @property {string} export_metadata.version - Export version
 * @property {string} export_metadata.source - Export source
 * @property {GraphConfig} [graph_config] - Graph visualization configuration
 */

// Export types for JSDoc validation
export const GraphTypes = {
  CIRCUIT: 'circuit',
  ATTENTION: 'attention',
  ACTIVATION_FLOW: 'activation_flow',
  MODEL_ARCHITECTURE: 'model_architecture',
  COMPARISON: 'comparison'
};

export const NodeTypes = {
  NEURON: 'neuron',
  ATTENTION_HEAD: 'attention_head',
  LAYER: 'layer',
  CIRCUIT: 'circuit',
  TOKEN: 'token',
  FEATURE: 'feature'
};

export const LinkTypes = {
  ACTIVATION: 'activation',
  ATTENTION: 'attention',
  CIRCUIT: 'circuit',
  CAUSAL: 'causal',
  SIMILARITY: 'similarity'
};

export const LayoutAlgorithms = {
  FORCE: 'force',
  HIERARCHICAL: 'hierarchical',
  CIRCULAR: 'circular'
};

export const Themes = {
  LIGHT: 'light',
  DARK: 'dark',
  CUSTOM: 'custom'
};