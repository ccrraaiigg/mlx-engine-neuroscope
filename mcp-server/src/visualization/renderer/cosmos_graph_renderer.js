/**
 * @fileoverview Cosmos Graph renderer for high-performance graph visualization
 * Provides WebGL-based rendering using the @cosmos.gl/graph library
 */

import { WebGLCapabilities, getCachedCapabilities } from '../utils/webgl_capabilities.js';

// Will be set when the module is loaded
let Graph = null;
const COSMOS_GRAPH_PATH = new URL('../../node_modules/@cosmos.gl/graph/dist/index.js', import.meta.url).href;

/**
 * High-performance WebGL-based graph renderer using Cosmos Graph
 */
export class CosmosGraphRenderer {
  /**
   * @param {HTMLElement} container - Container element for the visualization
   * @param {Partial<import('../types/graph_types.js').CosmosGraphConfig>} config - Cosmos Graph configuration
   */
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      width: 800,
      height: 600,
      backgroundColor: [0.1, 0.1, 0.1, 1],
      linkWidth: 2,
      linkColor: [1, 1, 1, 0.6],
      pointColor: [0.2, 0.6, 1, 1],
      pointSize: 6,
      enableNodeHover: true,
      enableNodeClick: true,
      simulationGravity: 0.1,
      simulationRepulsion: 0.8,
      linkDistance: 100,
      ...options
    };
    this.isInitialized = false;
    this._hoverHandlers = [];
    this._clickHandlers = [];
    this.currentGraphData = null;
    
    // Event handlers
    this.nodeClickHandlers = [];
    this.nodeHoverHandlers = [];
    this.linkClickHandlers = [];
    
    this._validateContainer();
    
    // Initialize capabilities
    this.capabilities = getCachedCapabilities();
    
    // Check compatibility but don't throw, just log a warning
    if (!this._checkCompatibility()) {
      console.warn('WebGL compatibility issues detected. Some features may not work as expected.');
    }
  }

  /**
   * Initialize the Cosmos Graph renderer
   * @returns {Promise<void>}
   */
  async initialize() {
    if (this.isInitialized) {
      return;
    }

    try {
      // Check WebGL support before importing
      const tempCanvas = document.createElement('canvas');
      const gl = tempCanvas.getContext('webgl2') || tempCanvas.getContext('webgl');
      if (!gl) {
        throw new Error('WebGL is not supported in this browser. Please try using a modern browser with WebGL support.');
      }

      // Dynamically import Cosmos Graph
      console.log('Importing Cosmos Graph from:', COSMOS_GRAPH_PATH);
      const cosmosModule = await import(/* @vite-ignore */ COSMOS_GRAPH_PATH);
      Graph = cosmosModule.Graph || cosmosModule.default || cosmosModule;
      
      if (!Graph) {
        console.error('Failed to load Cosmos Graph module:', cosmosModule);
        throw new Error('Failed to load Cosmos Graph module. Make sure it is installed.');
      }
      
      // Set up container styles (use the container directly like the working version)
      this.container.style.height = '100vh';
      this.container.style.width = '100%';
      
      if (!Graph) {
        throw new Error('Failed to load @cosmos.gl/graph module');
      }
      
      // Initialize graph with proper configuration (matching working version)
      const graphConfig = {
        spaceSize: 4096,
        backgroundColor: '#222222',
        pointColor: '#FF6B6B',
        pointSize: 15,
        linkColor: '#FFFFFF',
        linkWidth: 2,
        scalePointsOnZoom: true,
        simulationFriction: 0.1,
        simulationGravity: 0,
        simulationRepulsion: 0.5,
        enableDrag: true,
        onClick: (pointIndex) => {
          console.log('Clicked point index:', pointIndex);
          if (this.currentGraphData && pointIndex !== undefined) {
            const node = this.currentGraphData.nodes[pointIndex];
            if (node) {
              this.nodeClickHandlers.forEach(handler => handler(node));
            }
          }
        },
        onMouseMove: (pointIndex) => {
          const node = pointIndex !== undefined && this.currentGraphData 
            ? this.currentGraphData.nodes[pointIndex] 
            : null;
          this.nodeHoverHandlers.forEach(handler => handler(node));
        }
      };
      
      // Initialize the graph with container as first parameter (directly like working version)
      this.graph = new Graph(this.container, graphConfig);
      
      // Canvas will be created by Cosmos Graph automatically
      
      // Debug container
      console.log('Container dimensions:', {
        width: this.container.clientWidth,
        height: this.container.clientHeight,
        container: this.container
      });
      
      // Debug WebGL context
      const canvasElement = this.container.querySelector('canvas');
      if (canvasElement) {
        const gl = canvasElement.getContext('webgl2') || canvasElement.getContext('webgl');
        console.log('WebGL Context:', {
          canvas: canvasElement,
          context: gl,
          contextAttributes: gl ? gl.getContextAttributes() : null,
          extensions: gl ? gl.getSupportedExtensions() : []
        });
      }
      
      // Set up event listeners
      if (this._setupEventListeners) {
        this._setupEventListeners();
      }
      
      this.isInitialized = true;
      console.log('Cosmos Graph renderer initialized successfully');
    } catch (error) {
      console.error('Failed to initialize Cosmos Graph renderer:', error);
      throw new Error(`Cosmos Graph initialization failed: ${error.message}`);
    }
  }

  /**
   * Load graph data into the renderer
   * @param {import('../types/graph_types.js').GraphData} graphData
   * @returns {Promise<void>}
   */
  async loadGraph(graphData) {
    if (!this.isInitialized) await this.initialize();
    
    try {
      // Validate the graph data
      this._validateGraphData(graphData);
      
      // Process the graph data
      const processedGraphData = this._processGraphData(graphData);
      
      // Convert to Cosmos Graph format
      const cosmosData = this._convertToCosmosGraphFormat(processedGraphData);
      
      // Use the exact same approach that worked in the direct version
      console.log('Setting positions:', cosmosData.positions.slice(0, 10));
      console.log('Setting links:', cosmosData.links.slice(0, 10));

      this.graph.setPointPositions(cosmosData.positions);
      this.graph.setLinks(cosmosData.links);
      this.graph.render();
      
      setTimeout(() => {
        this.graph.fitView(1000, 0.2);
      }, 100);
      
      console.log('Graph data loaded successfully');
      
      // Store the current graph data for reference
      this.currentGraphData = processedGraphData;
      
    } catch (error) {
      console.error('Failed to load graph:', error);
      throw new Error(`Graph loading failed: ${error.message}`);
    }
  }

  /**
   * Update node data
   * @param {import('../types/graph_types.js').NodeUpdate[]} nodeUpdates
   */
  updateNodeData(nodeUpdates) {
    if (!this.isInitialized || !this.currentGraphData) {
      throw new Error('Renderer not initialized or no graph loaded');
    }

    try {
      // Update internal graph data
      nodeUpdates.forEach(update => {
        const nodeIndex = this.currentGraphData.nodes.findIndex(n => n.id === update.id);
        if (nodeIndex !== -1) {
          this.currentGraphData.nodes[nodeIndex] = {
            ...this.currentGraphData.nodes[nodeIndex],
            ...update.data
          };
        }
      });

      // Convert and update Cosmos Graph
      const cosmosData = this._convertToCosmosGraphFormat(this.currentGraphData);
      this.graph.setPointPositions(cosmosData.positions);
      this.graph.setPointSizes(cosmosData.sizes);
      this.graph.setPointColors(cosmosData.colors);
    } catch (error) {
      console.error('Failed to update node data:', error);
      throw new Error(`Node update failed: ${error.message}`);
    }
  }

  /**
   * Update link data
   * @param {import('../types/graph_types.js').LinkUpdate[]} linkUpdates
   */
  updateLinkData(linkUpdates) {
    if (!this.isInitialized || !this.currentGraphData) {
      throw new Error('Renderer not initialized or no graph loaded');
    }

    try {
      // Update internal graph data
      linkUpdates.forEach(update => {
        const linkIndex = this.currentGraphData.links.findIndex(l => l.id === update.id);
        if (linkIndex !== -1) {
          this.currentGraphData.links[linkIndex] = {
            ...this.currentGraphData.links[linkIndex],
            ...update.data
          };
        }
      });

      // Convert and update Cosmos Graph
      const cosmosData = this._convertToCosmosGraphFormat(this.currentGraphData);
      this.graph.setLinks(cosmosData.links);
    } catch (error) {
      console.error('Failed to update link data:', error);
      throw new Error(`Link update failed: ${error.message}`);
    }
  }

  /**
   * Export visualization as image
   * @param {'png'|'svg'} format - Export format
   * @returns {Promise<Blob>}
   */
  async exportImage(format = 'png') {
    if (!this.isInitialized) {
      throw new Error('Renderer not initialized');
    }

    try {
      if (format === 'png') {
        // Use Cosmos Graph's built-in export if available
        if (this.graph.exportImage) {
          return await this.graph.exportImage();
        } else {
          // Fallback to canvas export
          return this._exportCanvasAsPNG();
        }
      } else if (format === 'svg') {
        // Generate SVG fallback
        if (!this.currentGraphData) {
          throw new Error('No graph data loaded');
        }
        const svgString = WebGLCapabilities.provideFallbackVisualization(this.currentGraphData);
        return new Blob([svgString], { type: 'image/svg+xml' });
      } else {
        throw new Error(`Unsupported export format: ${format}`);
      }
    } catch (error) {
      console.error('Failed to export image:', error);
      throw new Error(`Image export failed: ${error.message}`);
    }
  }

  /**
   * Export graph data
   * @returns {import('../types/graph_types.js').GraphExportData}
   */
  exportData() {
    if (!this.currentGraphData) {
      throw new Error('No graph data loaded');
    }

    return {
      graph_data: this.currentGraphData,
      cosmos_config: this.options,
      export_metadata: {
        exported_at: new Date(),
        version: '1.0.0',
        source: 'CosmosGraphRenderer'
      }
    };
  }

  /**
   * Add node click handler
   * @param {function(import('../types/graph_types.js').NodeData): void} callback
   */
  onNodeClick(callback) {
    this.nodeClickHandlers.push(callback);
  }

  /**
   * Add node hover handler
   * @param {function(import('../types/graph_types.js').NodeData|null): void} callback
   */
  onNodeHover(callback) {
    this.nodeHoverHandlers.push(callback);
  }

  /**
   * Add link click handler
   * @param {function(import('../types/graph_types.js').LinkData): void} callback
   */
  onLinkClick(callback) {
    this.linkClickHandlers.push(callback);
  }

  /**
   * Highlight specific nodes
   * @param {string[]} nodeIds
   */
  highlightNodes(nodeIds) {
    if (!this.isInitialized || !this.currentGraphData) {
      return;
    }

    // Update node highlighting in the current graph data
    this.currentGraphData.nodes.forEach(node => {
      if (nodeIds.includes(node.id)) {
        node.metadata = node.metadata || {};
        node.metadata.highlighted = true;
      } else if (node.metadata) {
        node.metadata.highlighted = false;
      }
    });

    // Update visualization
    const cosmosData = this._convertToCosmosGraphFormat(this.currentGraphData);
    this.graph.setPointColors(cosmosData.colors);
  }

  /**
   * Dispose of the renderer and clean up resources
   */
  dispose() {
    if (this.graph) {
      if (this.graph.destroy) {
        this.graph.destroy();
      }
      this.graph = null;
    }
    
    this.nodeClickHandlers = [];
    this.nodeHoverHandlers = [];
    this.linkClickHandlers = [];
    this.currentGraphData = null;
    this.isInitialized = false;
  }

  /**
   * Merge user config with defaults and capability-based optimizations
   * @private
   * @param {Partial<import('../types/graph_types.js').CosmosGraphConfig>} userConfig
   * @returns {import('../types/graph_types.js').CosmosGraphConfig}
   */
  _mergeConfig(userConfig) {
    const capabilities = getCachedCapabilities();
    const optimalConfig = WebGLCapabilities.getOptimalConfig(capabilities);
    
    const defaultConfig = {
      pointSize: 4,
      linkWidth: 1,
      backgroundColor: '#0d1117', // Dark theme background
      pointColor: '#58a6ff', // GitHub blue for dark theme
      linkColor: '#30363d', // Dark theme link color
      showLabels: true,
      simulationFriction: 0.85,
      simulationGravity: 0.1,
      simulationRepulsion: 1.0,
      linkDistance: 50,
      ...optimalConfig
    };

    return { ...defaultConfig, ...userConfig };
  }

  /**
   * Validate container element
   * @private
   */
  _validateContainer() {
    if (!this.container || !(this.container instanceof HTMLElement)) {
      throw new Error('Invalid container element provided');
    }
  }

  /**
   * Check WebGL compatibility
   * @private
   */
  _checkCompatibility() {
    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      if (!gl) {
        console.warn('WebGL is not supported or disabled in this browser');
        return false;
      }
      return true;
    } catch (error) {
      console.warn('Error checking WebGL compatibility:', error);
      return false;
    }
  }

  /**
   * Set up event listeners for Cosmos Graph
   * @private
   */
  _setupEventListeners() {
    if (!this.graph) return;

    // Node click events
    if (this.graph.onClick) {
      this.graph.onClick((pointIndex) => {
        if (this.currentGraphData && pointIndex !== undefined) {
          const node = this.currentGraphData.nodes[pointIndex];
          if (node) {
            this.nodeClickHandlers.forEach(handler => handler(node));
          }
        }
      });
    }

    // Node hover events
    if (this.graph.onMouseMove) {
      this.graph.onMouseMove((pointIndex) => {
        const node = pointIndex !== undefined && this.currentGraphData 
          ? this.currentGraphData.nodes[pointIndex] 
          : null;
        this.nodeHoverHandlers.forEach(handler => handler(node));
      });
    }
  }

  /**
   * Validate graph data structure
   * @private
   * @param {import('../types/graph_types.js').GraphData} graphData
   */
  _validateGraphData(graphData) {
    if (!graphData || typeof graphData !== 'object') {
      throw new Error('Invalid graph data: must be an object');
    }

    if (!Array.isArray(graphData.nodes)) {
      throw new Error('Invalid graph data: nodes must be an array');
    }

    if (!Array.isArray(graphData.links)) {
      throw new Error('Invalid graph data: links must be an array');
    }

    // Validate node structure
    graphData.nodes.forEach((node, index) => {
      if (!node.id || typeof node.id !== 'string') {
        throw new Error(`Invalid node at index ${index}: id must be a string`);
      }
      if (typeof node.value !== 'number') {
        throw new Error(`Invalid node at index ${index}: value must be a number`);
      }
    });

    // Validate link structure
    const nodeIds = new Set(graphData.nodes.map(n => n.id));
    graphData.links.forEach((link, index) => {
      if (!nodeIds.has(link.source)) {
        throw new Error(`Invalid link at index ${index}: source node '${link.source}' not found`);
      }
      if (!nodeIds.has(link.target)) {
        throw new Error(`Invalid link at index ${index}: target node '${link.target}' not found`);
      }
      // Ensure weight is a valid number, default to 1 if not provided
      if (typeof link.weight !== 'number' && typeof link.value === 'number') {
        link.weight = link.value; // Use value if weight is not provided
      } else if (typeof link.weight !== 'number') {
        link.weight = 1; // Default weight if neither weight nor value is provided
      }
    });
  }

  /**
   * Process graph data for optimal rendering
   * @private
   * @param {import('../types/graph_types.js').GraphData} graphData
   * @returns {import('../types/graph_types.js').GraphData}
   */
  _processGraphData(graphData) {
    let processedData = { ...graphData };

    // Simplify graph for low-performance devices (if capabilities are available)
    if (this.capabilities?.performanceLevel === 'low' && graphData.nodes.length > 500) {
      processedData = WebGLCapabilities.simplifyGraph(graphData, 500);
    } else if (this.capabilities?.performanceLevel === 'medium' && graphData.nodes.length > 1000) {
      processedData = WebGLCapabilities.simplifyGraph(graphData, 1000);
    }

    return processedData;
  }

  /**
   * Convert graph data to Cosmos Graph format
   * @private
   * @param {import('../types/graph_types.js').GraphData} graphData
   * @returns {{positions: Float32Array, sizes: Float32Array, colors: Float32Array, links: Float32Array, linkColors: Float32Array, linkWidths: Float32Array}}
   */
  _convertToCosmosGraphFormat(graphData) {
    console.log('Converting graph data for rendering...');
    
    // Create a map of node IDs to indices
    const nodeIndexMap = new Map();
    
    // Prepare arrays for Cosmos Graph
    const positions = [];
    const sizes = [];
    const colors = [];
    
    // Process nodes
    graphData.nodes.forEach((node, index) => {
      nodeIndexMap.set(node.id, index);
      
      // Positions: [x1, y1, x2, y2, ...]
      // Scale down the positions to fit better in the viewport
      const x = node.position?.x ? node.position.x * 0.3 : Math.random() * 100 - 50;
      const y = node.position?.y ? node.position.y * 0.3 : Math.random() * 100 - 50;
      positions.push(x, y);
      
      // Sizes: [size1, size2, ...]
      sizes.push(node.size || 15);
      
      // Colors: [r1, g1, b1, a1, r2, g2, b2, a2, ...]
      const nodeColor = node.color || [1.0, 0.2, 0.4, 1.0]; // Bright pink default
      if (Array.isArray(nodeColor) && nodeColor.length >= 4) {
        colors.push(nodeColor[0], nodeColor[1], nodeColor[2], nodeColor[3]);
      } else {
        colors.push(1.0, 0.2, 0.4, 1.0); // Bright pink fallback
      }
    });

    // Process links
    const links = [];
    const linkColors = [];
    const linkWidths = [];
    
    graphData.links.forEach((link, index) => {
      const sourceIdx = nodeIndexMap.get(link.source);
      const targetIdx = nodeIndexMap.get(link.target);
      
      if (sourceIdx !== undefined && targetIdx !== undefined) {
        // Links: [source1, target1, source2, target2, ...]
        links.push(sourceIdx, targetIdx);
        
        // Link colors: [r1, g1, b1, a1, r2, g2, b2, a2, ...]
        const linkColor = link.color || this.options.linkColor || [1, 1, 1, 0.6];
        if (Array.isArray(linkColor)) {
          linkColors.push(...linkColor);
        } else {
          linkColors.push(1, 1, 1, 0.6); // Default white
        }
        
        // Link widths: [width1, width2, ...]
        linkWidths.push(link.width || this.options.linkWidth || 1);
      } else {
        console.warn(`Skipping invalid link: ${link.source} -> ${link.target}`);
      }
    });

    console.log(`Converted graph data: ${graphData.nodes.length} nodes, ${links.length / 2} links`);
    
    return {
      positions: new Float32Array(positions),
      sizes: new Float32Array(sizes),
      colors: new Float32Array(colors),
      links: new Float32Array(links),
      linkColors: new Float32Array(linkColors),
      linkWidths: new Float32Array(linkWidths)
    };
  }

  /**
   * Export canvas as PNG blob
   * @private
   * @returns {Promise<Blob>}
   */
  async _exportCanvasAsPNG() {
    return new Promise((resolve, reject) => {
      const canvasElement = this.container.querySelector('canvas');
      if (!canvasElement) {
        reject(new Error('Canvas not found'));
        return;
      }

      canvasElement.toBlob((blob) => {
        if (blob) {
          resolve(blob);
        } else {
          reject(new Error('Failed to create PNG blob'));
        }
      }, 'image/png');
    });
  }
}
