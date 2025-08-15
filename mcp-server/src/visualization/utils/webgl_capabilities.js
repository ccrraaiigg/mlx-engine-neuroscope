/**
 * @fileoverview WebGL capability detection and performance optimization utilities
 * Provides comprehensive WebGL feature detection and fallback mechanisms
 */

/**
 * WebGL capabilities detection and optimization utilities
 */
export class WebGLCapabilities {
  /**
   * Detect comprehensive WebGL capabilities
   * @returns {Object} Capabilities object with detailed WebGL information
   */
  static detectCapabilities() {
    const capabilities = {
      webgl: false,
      webgl2: false,
      maxTextureSize: 0,
      maxVertexUniforms: 0,
      maxFragmentUniforms: 0,
      maxVaryingVectors: 0,
      maxVertexAttribs: 0,
      maxViewportDims: [0, 0],
      extensions: [],
      performanceLevel: 'low',
      renderer: 'unknown',
      vendor: 'unknown'
    };

    try {
      // Create a test canvas
      const canvas = document.createElement('canvas');
      canvas.width = 1;
      canvas.height = 1;

      // Test WebGL 2.0 first
      let gl = canvas.getContext('webgl2');
      if (gl) {
        capabilities.webgl2 = true;
        capabilities.webgl = true;
        console.log('WebGL2 context created successfully');
      } else {
        // Fallback to WebGL 1.0
        gl = canvas.getContext('webgl') || canvas.getContext('experimental-webgl');
        if (gl) {
          capabilities.webgl = true;
          console.log('WebGL1 context created successfully');
        }
      }

      if (!gl) {
        console.warn('No WebGL support detected');
        return capabilities;
      }

      // Get basic parameters
      capabilities.maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE);
      capabilities.maxVertexUniforms = gl.getParameter(gl.MAX_VERTEX_UNIFORM_VECTORS);
      capabilities.maxFragmentUniforms = gl.getParameter(gl.MAX_FRAGMENT_UNIFORM_VECTORS);
      capabilities.maxVaryingVectors = gl.getParameter(gl.MAX_VARYING_VECTORS);
      capabilities.maxVertexAttribs = gl.getParameter(gl.MAX_VERTEX_ATTRIBS);
      capabilities.maxViewportDims = gl.getParameter(gl.MAX_VIEWPORT_DIMS);

      // Get renderer info
      const debugInfo = gl.getExtension('WEBGL_debug_renderer_info');
      if (debugInfo) {
        capabilities.renderer = gl.getParameter(debugInfo.UNMASKED_RENDERER_WEBGL);
        capabilities.vendor = gl.getParameter(debugInfo.UNMASKED_VENDOR_WEBGL);
      }

      // Get supported extensions
      capabilities.extensions = gl.getSupportedExtensions() || [];

      // Determine performance level based on capabilities
      capabilities.performanceLevel = this._determinePerformanceLevel(capabilities);

      console.log('WebGL capabilities detected:', capabilities);
      return capabilities;

    } catch (error) {
      console.error('Error detecting WebGL capabilities:', error);
      return capabilities;
    }
  }

  /**
   * Determine performance level based on detected capabilities
   * @private
   * @param {Object} capabilities - Detected capabilities
   * @returns {string} Performance level: 'high', 'medium', or 'low'
   */
  static _determinePerformanceLevel(capabilities) {
    let score = 0;

    // WebGL2 support adds significant points
    if (capabilities.webgl2) score += 30;
    else if (capabilities.webgl) score += 15;

    // Texture size scoring
    if (capabilities.maxTextureSize >= 16384) score += 20;
    else if (capabilities.maxTextureSize >= 8192) score += 15;
    else if (capabilities.maxTextureSize >= 4096) score += 10;
    else if (capabilities.maxTextureSize >= 2048) score += 5;

    // Uniform support scoring
    if (capabilities.maxVertexUniforms >= 256) score += 15;
    else if (capabilities.maxVertexUniforms >= 128) score += 10;
    else if (capabilities.maxVertexUniforms >= 64) score += 5;

    // Extension support (key extensions for performance)
    const performanceExtensions = [
      'OES_vertex_array_object',
      'WEBGL_draw_buffers',
      'OES_texture_float',
      'OES_element_index_uint',
      'ANGLE_instanced_arrays'
    ];

    const supportedPerfExtensions = performanceExtensions.filter(ext => 
      capabilities.extensions.includes(ext)
    );
    score += supportedPerfExtensions.length * 3;

    // GPU detection (rough heuristics based on renderer string)
    if (capabilities.renderer) {
      const renderer = capabilities.renderer.toLowerCase();
      if (renderer.includes('nvidia') || renderer.includes('amd') || renderer.includes('radeon')) {
        score += 10; // Dedicated GPU
      } else if (renderer.includes('intel') && (renderer.includes('iris') || renderer.includes('uhd'))) {
        score += 5; // Modern integrated GPU
      } else if (renderer.includes('apple') || renderer.includes('m1') || renderer.includes('m2')) {
        score += 15; // Apple Silicon
      }
    }

    // Determine level based on score
    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  }

  /**
   * Get optimal configuration based on detected capabilities
   * @param {Object} capabilities - Detected capabilities object
   * @returns {Object} Optimal configuration settings
   */
  static getOptimalConfig(capabilities) {
    const config = {
      pointSize: 4,
      linkWidth: 1,
      maxNodes: 1000,
      maxLinks: 2000,
      enableAntialiasing: false,
      enableShadows: false,
      lodEnabled: false,
      animationQuality: 'medium'
    };

    switch (capabilities.performanceLevel) {
      case 'high':
        config.pointSize = 6;
        config.linkWidth = 2;
        config.maxNodes = 5000;
        config.maxLinks = 10000;
        config.enableAntialiasing = true;
        config.enableShadows = capabilities.webgl2;
        config.lodEnabled = true;
        config.animationQuality = 'high';
        break;

      case 'medium':
        config.pointSize = 5;
        config.linkWidth = 1.5;
        config.maxNodes = 2000;
        config.maxLinks = 4000;
        config.enableAntialiasing = capabilities.webgl2;
        config.lodEnabled = true;
        config.animationQuality = 'medium';
        break;

      case 'low':
        config.pointSize = 3;
        config.linkWidth = 1;
        config.maxNodes = 500;
        config.maxLinks = 1000;
        config.animationQuality = 'low';
        break;
    }

    return config;
  }

  /**
   * Simplify graph data for performance on lower-end devices
   * @param {Object} graphData - Original graph data
   * @param {number} maxNodes - Maximum number of nodes to keep
   * @returns {Object} Simplified graph data
   */
  static simplifyGraph(graphData, maxNodes = 500) {
    if (!graphData.nodes || graphData.nodes.length <= maxNodes) {
      return graphData;
    }

    console.log(`Simplifying graph from ${graphData.nodes.length} to ${maxNodes} nodes`);

    // Sort nodes by importance (value) and keep the most important ones
    const sortedNodes = [...graphData.nodes].sort((a, b) => (b.value || 0) - (a.value || 0));
    const keptNodes = sortedNodes.slice(0, maxNodes);
    const keptNodeIds = new Set(keptNodes.map(n => n.id));

    // Filter links to only include connections between kept nodes
    const keptLinks = graphData.links.filter(link => 
      keptNodeIds.has(link.source) && keptNodeIds.has(link.target)
    );

    return {
      ...graphData,
      nodes: keptNodes,
      links: keptLinks,
      metadata: {
        ...graphData.metadata,
        simplified: true,
        originalNodeCount: graphData.nodes.length,
        originalLinkCount: graphData.links.length,
        simplificationRatio: maxNodes / graphData.nodes.length
      }
    };
  }

  /**
   * Provide fallback visualization for unsupported browsers
   * @param {Object} graphData - Graph data to visualize
   * @returns {string} SVG string for fallback visualization
   */
  static provideFallbackVisualization(graphData) {
    const width = 800;
    const height = 600;
    const centerX = width / 2;
    const centerY = height / 2;

    let svg = `<svg width="${width}" height="${height}" xmlns="http://www.w3.org/2000/svg">`;
    svg += `<rect width="100%" height="100%" fill="#0d1117"/>`;

    // Draw links first (so they appear behind nodes)
    if (graphData.links) {
      graphData.links.forEach(link => {
        const sourceNode = graphData.nodes.find(n => n.id === link.source);
        const targetNode = graphData.nodes.find(n => n.id === link.target);
        
        if (sourceNode && targetNode) {
          const x1 = (sourceNode.position?.x || 0) + centerX;
          const y1 = (sourceNode.position?.y || 0) + centerY;
          const x2 = (targetNode.position?.x || 0) + centerX;
          const y2 = (targetNode.position?.y || 0) + centerY;
          
          svg += `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#30363d" stroke-width="1" opacity="0.6"/>`;
        }
      });
    }

    // Draw nodes
    if (graphData.nodes) {
      graphData.nodes.forEach(node => {
        const x = (node.position?.x || 0) + centerX;
        const y = (node.position?.y || 0) + centerY;
        const r = (node.size || 5);
        const color = node.color || '#58a6ff';
        
        svg += `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="0.8"/>`;
        
        if (node.label) {
          svg += `<text x="${x}" y="${y + r + 15}" text-anchor="middle" fill="#c9d1d9" font-size="10" font-family="Arial">${node.label}</text>`;
        }
      });
    }

    // Add title
    svg += `<text x="${centerX}" y="30" text-anchor="middle" fill="#58a6ff" font-size="16" font-family="Arial">Graph Visualization (SVG Fallback)</text>`;
    
    svg += '</svg>';
    return svg;
  }

  /**
   * Test WebGL context creation and basic functionality
   * @returns {Object} Test results
   */
  static runWebGLTests() {
    const results = {
      contextCreation: false,
      shaderCompilation: false,
      bufferCreation: false,
      textureCreation: false,
      errors: []
    };

    try {
      const canvas = document.createElement('canvas');
      const gl = canvas.getContext('webgl2') || canvas.getContext('webgl');
      
      if (!gl) {
        results.errors.push('Failed to create WebGL context');
        return results;
      }
      
      results.contextCreation = true;

      // Test shader compilation
      const vertexShader = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vertexShader, `
        attribute vec2 position;
        void main() {
          gl_Position = vec4(position, 0.0, 1.0);
        }
      `);
      gl.compileShader(vertexShader);
      
      if (gl.getShaderParameter(vertexShader, gl.COMPILE_STATUS)) {
        results.shaderCompilation = true;
      } else {
        results.errors.push('Shader compilation failed');
      }

      // Test buffer creation
      const buffer = gl.createBuffer();
      if (buffer) {
        results.bufferCreation = true;
      } else {
        results.errors.push('Buffer creation failed');
      }

      // Test texture creation
      const texture = gl.createTexture();
      if (texture) {
        results.textureCreation = true;
      } else {
        results.errors.push('Texture creation failed');
      }

    } catch (error) {
      results.errors.push(`WebGL test error: ${error.message}`);
    }

    return results;
  }
}

/**
 * Cached capabilities to avoid repeated detection
 */
let cachedCapabilities = null;

/**
 * Get cached capabilities or detect them if not cached
 * @returns {Object} WebGL capabilities
 */
export function getCachedCapabilities() {
  if (!cachedCapabilities) {
    cachedCapabilities = WebGLCapabilities.detectCapabilities();
  }
  return cachedCapabilities;
}

/**
 * Clear cached capabilities (useful for testing)
 */
export function clearCapabilitiesCache() {
  cachedCapabilities = null;
}