/**
 * @fileoverview 3D Force Graph renderer for high-performance graph visualization
 * Provides WebGL-based rendering using the 3d-force-graph library
 */

import { WebGLCapabilities, getCachedCapabilities } from '../utils/webgl_capabilities.js';

// Will be set when the module is loaded
let ForceGraph3D = null;
let THREE = null;
const FORCE_GRAPH_3D_PATH = new URL('../../node_modules/3d-force-graph/dist/3d-force-graph.min.js', import.meta.url).href;
const THREE_PATH = new URL('../../node_modules/three/build/three.min.js', import.meta.url).href;

/**
 * High-performance WebGL-based graph renderer using 3D Force Graph
 */
export class ForceGraph3DRenderer {
  /**
   * @param {HTMLElement} container - Container element for the visualization
   * @param {Object} options - 3D Force Graph configuration options
   */
  constructor(container, options = {}) {
    this.container = container;
    this.options = {
      backgroundColor: '#0d1117',
      nodeColor: '#58a6ff',
      linkColor: '#30363d',
      nodeOpacity: 0.8,
      linkOpacity: 0.6,
      nodeRelSize: 4,
      linkWidth: 1.5,
      showNodeLabels: true,
      showLinkLabels: false,
      controlType: 'trackball',
      enableNodeDrag: true,
      enableNavigationControls: true,
      enablePointerInteraction: true,
      ...options
    };
    this.isInitialized = false;
    this.currentGraphData = null;
    this.graph = null;
    
    // Event handlers
    this.nodeClickHandlers = [];
    this.nodeHoverHandlers = [];
    this.linkClickHandlers = [];
    
    // Label tracking
    this.labelContainer = null;
    this.nodeLabels = new Map();
    
    // Movement tracking
    this.nodeMovementHistory = new Map();
    this.dragStartPositions = new Map();
    this.lastNodePositions = new Map();
    
    this._validateContainer();
    
    // Initialize capabilities
    this.capabilities = getCachedCapabilities();
    
    // Check compatibility but don't throw, just log a warning
    if (!this._checkCompatibility()) {
      console.warn('WebGL compatibility issues detected. Some features may not work as expected.');
    }
  }

  /**
   * Initialize the 3D Force Graph renderer
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

      // Load THREE.js first
      console.log('Loading THREE.js from:', THREE_PATH);
      
      const threeScript = document.createElement('script');
      threeScript.src = THREE_PATH;
      threeScript.type = 'text/javascript';
      
      await new Promise((resolve, reject) => {
        threeScript.onload = () => {
          console.log('THREE.js loaded successfully');
          THREE = window.THREE;
          if (!THREE) {
            reject(new Error('THREE.js not available on window after load'));
            return;
          }
          resolve();
        };
        threeScript.onerror = (error) => {
          console.error('Failed to load THREE.js:', error);
          reject(new Error('Failed to load THREE.js'));
        };
        document.head.appendChild(threeScript);
      });
      
      // Now load 3D Force Graph
      console.log('Loading 3D Force Graph UMD from:', FORCE_GRAPH_3D_PATH);
      
      const forceGraphScript = document.createElement('script');
      forceGraphScript.src = FORCE_GRAPH_3D_PATH;
      forceGraphScript.type = 'text/javascript';
      
      await new Promise((resolve, reject) => {
        forceGraphScript.onload = () => {
          console.log('3D Force Graph UMD script loaded successfully');
          resolve();
        };
        forceGraphScript.onerror = (error) => {
          console.error('Failed to load 3D Force Graph UMD script:', error);
          reject(new Error('Failed to load 3D Force Graph UMD script'));
        };
        document.head.appendChild(forceGraphScript);
      });
      
      // Both should now be available on the window object
      ForceGraph3D = window.ForceGraph3D;
      
      if (!ForceGraph3D || typeof ForceGraph3D !== 'function') {
        console.error('3D Force Graph not found on window object');
        console.error('Available on window:', Object.keys(window).filter(k => k.includes('Force')));
        throw new Error('3D Force Graph constructor not found after UMD load');
      }
      
      if (!THREE) {
        console.error('THREE.js not found on window object');
        throw new Error('THREE.js not available after script load');
      }
      
      console.log('3D Force Graph constructor loaded successfully:', ForceGraph3D);
      
      // Set up container styles
      this.container.style.height = '100vh';
      this.container.style.width = '100%';
      
      // Initialize 3D Force Graph
      this.graph = new ForceGraph3D(this.container);
      
      // Configure the graph with user options
      this.graph
        .backgroundColor(this.options.backgroundColor)
        .nodeRelSize(this.options.nodeRelSize)
        .linkWidth(this.options.linkWidth)
        .enableNodeDrag(this.options.enableNodeDrag);

      // Set up node appearance with always-visible labels
      this.graph
        .nodeLabel('') // Disable built-in tooltips completely
        .nodeColor(node => node.color || this.options.nodeColor)
        .nodeVal(node => node.value || 1)
        .showNavInfo(false) // Hide navigation info to make more room for labels
        .nodeAutoColorBy('color') // Use the color property for auto-coloring
        .linkHoverPrecision(2); // Make links easier to hover over

      // Use nodeThreeObjectExtend to add persistent text labels
      this.graph
        .nodeThreeObjectExtend(true)
        .nodeThreeObject(node => {
          try {
            // Create a text sprite that will always be visible
            const sprite = this._createTextSprite(node.label || node.id, {
              fontSize: 32,
              fontColor: '#ffffff',
              backgroundColor: 'rgba(0, 0, 0, 0.8)',
              borderColor: '#58a6ff',
              borderWidth: 2
            });
            
            // Position the sprite very close to the node
            const nodeSize = (node.value || 1) * this.options.nodeRelSize;
            sprite.position.set(0, nodeSize + 3, 0); // Very close, using relative positioning
            
            // Store reference for movement tracking and ensure it persists
            sprite.userData = { 
              nodeId: node.id, 
              isLabel: true,
              isPersistent: true // Mark as persistent
            };
            
            // Make sure the sprite is always rendered
            sprite.renderOrder = 999; // High render order to stay on top
            sprite.material.depthTest = false;
            sprite.material.depthWrite = false;
            
            console.log('Created persistent text sprite for node:', node.id);
            return sprite;
            
          } catch (error) {
            console.log('Error creating text sprite for node:', node.id, error);
            return null;
          }
        });

      // Set up link appearance  
      this.graph
        .linkLabel(link => link.label || '')
        .linkColor(link => link.color || this.options.linkColor)
        .linkWidth(link => link.width || this.options.linkWidth);

      // Set up event handlers
      this.graph
        .onNodeClick((node, event) => {
          console.log('Clicked node:', node);
          this.nodeClickHandlers.forEach(handler => handler(node, event));
        })
        .onNodeHover((node, prevNode) => {
          this.nodeHoverHandlers.forEach(handler => handler(node, prevNode));
        })
        .onLinkClick((link, event) => {
          console.log('Clicked link:', link);
          this.linkClickHandlers.forEach(handler => handler(link, event));
        })
        .onNodeDrag((node) => {
          const timestamp = new Date().toISOString();
          console.log(`[${timestamp}] Node ${node.id} being dragged:`, {
            x: node.x?.toFixed(2),
            y: node.y?.toFixed(2),
            z: node.z?.toFixed(2),
            velocity: this._calculateNodeVelocity(node)
          });
          
          // Labels should move automatically with nodeThreeObjectExtend
          // Just ensure visibility is maintained
          this._ensureLabelVisibility(node);
          
          // Track movement history for analysis
          this._trackNodeMovement(node, 'drag');
        })
        .onNodeDragEnd((node) => {
          const timestamp = new Date().toISOString();
          console.log(`[${timestamp}] Node ${node.id} drag ended at:`, {
            final_x: node.x?.toFixed(2),
            final_y: node.y?.toFixed(2),
            final_z: node.z?.toFixed(2),
            total_distance: this._getTotalDragDistance(node.id)
          });
          
          // Store the final position in the node data
          if (this.currentGraphData) {
            const nodeIndex = this.currentGraphData.nodes.findIndex(n => n.id === node.id);
            if (nodeIndex !== -1) {
              this.currentGraphData.nodes[nodeIndex].x = node.x;
              this.currentGraphData.nodes[nodeIndex].y = node.y;
              this.currentGraphData.nodes[nodeIndex].z = node.z;
              
              // Store position history
              if (!this.currentGraphData.nodes[nodeIndex].positionHistory) {
                this.currentGraphData.nodes[nodeIndex].positionHistory = [];
              }
              this.currentGraphData.nodes[nodeIndex].positionHistory.push({
                timestamp: timestamp,
                x: node.x,
                y: node.y,
                z: node.z,
                action: 'drag_end'
              });
            }
          }
          
          // Ensure label is visible after drag
          this._ensureLabelVisibility(node);
          
          // Clear movement tracking for this node
          this._clearNodeMovementTracking(node.id);
        });
      
      this.isInitialized = true;
      console.log('3D Force Graph renderer initialized successfully');
    } catch (error) {
      console.error('Failed to initialize 3D Force Graph renderer:', error);
      throw new Error(`3D Force Graph initialization failed: ${error.message}`);
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
      
      // Convert to 3D Force Graph format
      const forceGraphData = this._convertToForceGraphFormat(processedGraphData);
      
      console.log('Loading graph data:', {
        nodes: forceGraphData.nodes.length,
        links: forceGraphData.links.length
      });

      // Load data into 3D Force Graph
      this.graph.graphData(forceGraphData);
      
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

      // Convert and reload data in 3D Force Graph
      const forceGraphData = this._convertToForceGraphFormat(this.currentGraphData);
      this.graph.graphData(forceGraphData);
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

      // Convert and reload data in 3D Force Graph
      const forceGraphData = this._convertToForceGraphFormat(this.currentGraphData);
      this.graph.graphData(forceGraphData);
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
        source: 'ForceGraph3DRenderer'
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
        // Make highlighted nodes brighter
        node.color = '#ffff00'; // Yellow for highlighted nodes
      } else if (node.metadata) {
        node.metadata.highlighted = false;
        node.color = node.originalColor || this.options.nodeColor;
      }
    });

    // Update visualization
    const forceGraphData = this._convertToForceGraphFormat(this.currentGraphData);
    this.graph.graphData(forceGraphData);
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
   * Convert graph data to 3D Force Graph format
   * @private
   * @param {import('../types/graph_types.js').GraphData} graphData
   * @returns {{nodes: Array, links: Array}}
   */
  _convertToForceGraphFormat(graphData) {
    console.log('Converting graph data for 3D Force Graph rendering...');
    
    // Process nodes for 3D Force Graph format
    const nodes = graphData.nodes.map(node => ({
      id: node.id,
      label: node.label || node.id,
      value: node.value || 1,
      color: node.color || this.options.nodeColor,
      x: node.position?.x,
      y: node.position?.y,
      z: node.position?.z,
      ...node // Include any additional properties
    }));
    
    // Process links for 3D Force Graph format
    const links = graphData.links.map(link => ({
      source: link.source,
      target: link.target,
      label: link.label || '',
      value: link.weight || link.value || 1,
      color: link.color || this.options.linkColor,
      width: link.width || this.options.linkWidth,
      ...link // Include any additional properties
    }));

    console.log(`Converted graph data: ${nodes.length} nodes, ${links.length} links`);
    
    return { nodes, links };
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

  /**
   * Create a text sprite for node labels
   * @private
   * @param {string} text - Text to display
   * @param {Object} options - Styling options
   * @returns {THREE.Sprite}
   */
  _createTextSprite(text, options = {}) {
    const defaults = {
      fontSize: 24,
      fontColor: '#ffffff',
      backgroundColor: 'rgba(0, 0, 0, 0.7)',
      borderColor: '#58a6ff',
      borderWidth: 2,
      padding: 8
    };
    const config = { ...defaults, ...options };
    
    // Create canvas for text rendering
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');
    
    // Set font and measure text
    context.font = `${config.fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
    const textMetrics = context.measureText(text);
    const textWidth = textMetrics.width;
    const textHeight = config.fontSize;
    
    // Set canvas size with padding and border
    const padding = config.padding;
    const borderWidth = config.borderWidth;
    canvas.width = textWidth + (padding * 2) + (borderWidth * 2);
    canvas.height = textHeight + (padding * 2) + (borderWidth * 2);
    
    // Set font again (canvas resize clears context)
    context.font = `${config.fontSize}px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif`;
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    
    // Draw background
    if (config.backgroundColor !== 'transparent') {
      context.fillStyle = config.backgroundColor;
      context.fillRect(0, 0, canvas.width, canvas.height);
    }
    
    // Draw border
    if (config.borderWidth > 0) {
      context.strokeStyle = config.borderColor;
      context.lineWidth = config.borderWidth;
      context.strokeRect(
        borderWidth / 2, 
        borderWidth / 2, 
        canvas.width - borderWidth, 
        canvas.height - borderWidth
      );
    }
    
    // Draw text
    context.fillStyle = config.fontColor;
    context.fillText(
      text, 
      canvas.width / 2, 
      canvas.height / 2
    );
    
    // Create texture and sprite
    const texture = new THREE.CanvasTexture(canvas);
    texture.needsUpdate = true;
    
    const spriteMaterial = new THREE.SpriteMaterial({ 
      map: texture, 
      transparent: true,
      alphaTest: 0.1,
      depthTest: false,
      depthWrite: false
    });
    
    const sprite = new THREE.Sprite(spriteMaterial);
    
    // Scale sprite to larger size for better visibility
    const spriteScale = 0.8;
    sprite.scale.set(
      (canvas.width / 8) * spriteScale, 
      (canvas.height / 8) * spriteScale, 
      1
    );
    
    return sprite;
  }

  /**
   * Update node label position during movement
   * @private
   * @param {Object} node - Node being moved
   */
  _updateNodeLabelPosition(node) {
    try {
      if (!this.graph || !this.graph.scene) return;
      
      // Since we're using nodeThreeObjectExtend, the labels should move automatically
      // with their parent nodes. We just need to ensure they maintain their relative position.
      const scene = this.graph.scene();
      scene.traverse((child) => {
        if (child.userData && child.userData.nodeId === node.id && child.userData.isLabel) {
          // Maintain relative position - don't set absolute world coordinates
          const nodeSize = (node.value || 1) * this.options.nodeRelSize;
          child.position.set(0, nodeSize + 3, 0); // Keep relative to parent node
          
          // Ensure the label remains visible
          child.visible = true;
          if (child.material) {
            child.material.opacity = 1.0;
            child.material.transparent = true;
          }
        }
      });
    } catch (error) {
      console.log('Error updating label position for node:', node.id, error);
    }
  }

  /**
   * Ensure label visibility is maintained
   * @private
   * @param {Object} node - Node to check
   */
  _ensureLabelVisibility(node) {
    try {
      if (!this.graph || !this.graph.scene) return;
      
      const scene = this.graph.scene();
      let labelFound = false;
      
      scene.traverse((child) => {
        if (child.userData && child.userData.nodeId === node.id && child.userData.isLabel) {
          labelFound = true;
          
          // Force visibility
          child.visible = true;
          child.frustumCulled = false; // Don't cull based on camera frustum
          
          if (child.material) {
            child.material.opacity = 1.0;
            child.material.transparent = true;
            child.material.needsUpdate = true;
          }
          
          // Ensure proper render order
          child.renderOrder = 999;
        }
      });
      
      // If label not found, it might have been lost - could recreate here if needed
      if (!labelFound) {
        console.warn(`Label for node ${node.id} not found during visibility check`);
      }
      
    } catch (error) {
      console.log('Error ensuring label visibility for node:', node.id, error);
    }
  }

  /**
   * Track node movement history
   * @private
   * @param {Object} node - Node being moved
   * @param {string} action - Type of movement (drag, simulation, etc.)
   */
  _trackNodeMovement(node, action) {
    const nodeId = node.id;
    const timestamp = Date.now();
    
    // Initialize tracking for this node if not exists
    if (!this.nodeMovementHistory.has(nodeId)) {
      this.nodeMovementHistory.set(nodeId, []);
      this.dragStartPositions.set(nodeId, { x: node.x, y: node.y, z: node.z });
    }
    
    // Store current position
    const currentPosition = { x: node.x, y: node.y, z: node.z };
    const lastPosition = this.lastNodePositions.get(nodeId);
    
    // Calculate distance moved since last position
    let distanceMoved = 0;
    if (lastPosition) {
      distanceMoved = Math.sqrt(
        Math.pow(currentPosition.x - lastPosition.x, 2) +
        Math.pow(currentPosition.y - lastPosition.y, 2) +
        Math.pow(currentPosition.z - lastPosition.z, 2)
      );
    }
    
    // Add to movement history
    const movementHistory = this.nodeMovementHistory.get(nodeId);
    movementHistory.push({
      timestamp,
      position: currentPosition,
      action,
      distanceMoved: distanceMoved.toFixed(2)
    });
    
    // Keep only recent history (last 100 movements)
    if (movementHistory.length > 100) {
      movementHistory.splice(0, movementHistory.length - 100);
    }
    
    // Update last position
    this.lastNodePositions.set(nodeId, currentPosition);
  }

  /**
   * Calculate node velocity
   * @private
   * @param {Object} node - Node object
   * @returns {Object} Velocity data
   */
  _calculateNodeVelocity(node) {
    const nodeId = node.id;
    const movementHistory = this.nodeMovementHistory.get(nodeId);
    
    if (!movementHistory || movementHistory.length < 2) {
      return { speed: 0, direction: null };
    }
    
    const current = movementHistory[movementHistory.length - 1];
    const previous = movementHistory[movementHistory.length - 2];
    
    const timeDelta = (current.timestamp - previous.timestamp) / 1000; // seconds
    const distance = parseFloat(current.distanceMoved);
    const speed = timeDelta > 0 ? (distance / timeDelta).toFixed(2) : 0;
    
    // Calculate direction vector
    const dx = current.position.x - previous.position.x;
    const dy = current.position.y - previous.position.y;
    const dz = current.position.z - previous.position.z;
    
    return {
      speed: parseFloat(speed),
      direction: distance > 0 ? { x: dx, y: dy, z: dz } : null
    };
  }

  /**
   * Get total distance dragged for a node
   * @private
   * @param {string} nodeId - Node ID
   * @returns {number} Total distance
   */
  _getTotalDragDistance(nodeId) {
    const startPosition = this.dragStartPositions.get(nodeId);
    const movementHistory = this.nodeMovementHistory.get(nodeId);
    
    if (!startPosition || !movementHistory || movementHistory.length === 0) {
      return 0;
    }
    
    const totalDistance = movementHistory.reduce((sum, movement) => {
      return sum + parseFloat(movement.distanceMoved);
    }, 0);
    
    return totalDistance.toFixed(2);
  }

  /**
   * Clear movement tracking for a node
   * @private
   * @param {string} nodeId - Node ID
   */
  _clearNodeMovementTracking(nodeId) {
    this.nodeMovementHistory.delete(nodeId);
    this.dragStartPositions.delete(nodeId);
    this.lastNodePositions.delete(nodeId);
  }

}
