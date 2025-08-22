# Implementation Plan

- [x] 1. Set up 3d-force-graph integration foundation
  - Install 3d-force-graph JavaScript library and Three.js dependencies
  - Create JavaScript data structures for 3D graph data (NodeData with x,y,z, LinkData with particles, Graph3DData)
  - Set up WebGL and WebGL2 capability detection with Three.js compatibility testing
  - Create Node.js test server for development and testing of 3d-force-graph integration
  - _Requirements: 1.1, 4.1, 4.2_

- [ ] 2. Implement core graph data conversion
  - [ ] 2.1 Create Graph3DConverter class for MLX data transformation
    - Write conversion methods for circuit data to 3d-force-graph node/link format with 3D positioning
    - Implement activation strength to 3D visual properties mapping (size, color, particles)
    - Create calculate3DPositions method for layered, spherical, and hierarchical 3D layouts
    - Create unit tests for 3D data conversion accuracy
    - _Requirements: 1.1, 1.2_

  - [ ] 2.2 Implement Visualization3DGenerator service
    - Write generateCircuitGraph method that calls MLX Engine REST API and adds 3D positioning
    - Implement generateAttentionGraph for 3D attention pattern visualization with layer separation
    - Create generateActivationFlowGraph for 3D token-level activation flows with temporal positioning
    - Write unit tests for each 3D generation method
    - _Requirements: 1.1, 2.1, 2.2_

- [x] 3. Build 3d-force-graph renderer component
  - [x] 3.1 Create ForceGraph3DRenderer class with WebGL and Three.js integration
    - Implement 3d-force-graph initialization with performance-optimized 3D configuration
    - Write loadGraph method for rendering 3D graph data with smooth 3D animations
    - Create updateNodeData and updateLinkData methods for real-time 3D updates
    - Implement setCameraPosition and animateToNode methods for 3D navigation
    - Write unit tests for 3D rendering functionality
    - _Requirements: 1.1, 4.1, 4.3_

  - [x] 3.2 Implement InteractionController3D for 3D user interactions
    - Write 3D node click handlers that display detailed component information with 3D tooltips
    - Implement 3D hover functionality showing activation weights and metadata in 3D space
    - Create 3D node highlighting and selection mechanisms with camera focus
    - Implement 3D camera controls (trackball, orbit, fly) and background click handling
    - Write integration tests for 3D interaction behaviors
    - _Requirements: 1.3, 2.3_

- [ ] 4. Create MCP server visualization tools
  - [ ] 4.1 Implement core 3D visualization MCP tools
    - Write viz_create_circuit_graph tool with comprehensive JSON schema for 3D parameters
    - Implement viz_create_attention_graph tool for 3D attention head visualization with layer positioning
    - Create viz_create_flow_graph tool for 3D activation flow analysis with temporal positioning
    - Add viz_set_camera and viz_animate_camera tools for 3D camera control
    - Write unit tests for each 3D MCP tool with schema validation
    - _Requirements: 8.1, 8.2, 8.3_

  - [ ] 4.2 Build 3D export and sharing MCP tools
    - Implement viz_export_graph tool supporting screenshots, 3D models (GLB/GLTF), and JSON formats
    - Write viz_embed_graph tool generating embeddable 3D HTML code with Three.js
    - Create viz_update_graph tool for real-time 3D graph modifications
    - Write integration tests for 3D export functionality
    - _Requirements: 5.1, 5.2, 5.3_

- [ ] 5. Integrate with MLX Engine REST API
  - [ ] 5.1 Extend MLX Engine with 3D visualization endpoints
    - Add POST /visualizations/create endpoint for 3D graph generation with camera parameters
    - Implement GET /visualizations/{graph_id} for 3D graph retrieval with camera state
    - Create POST /visualizations/{graph_id}/export for 3D format conversion (screenshot, 3D model)
    - Write API integration tests with 3D activation data
    - _Requirements: 3.1, 3.2_

  - [ ] 5.2 Create ActivationCaptureClient for 3D visualization data
    - Write methods to fetch circuit activation data from MLX Engine with 3D positioning
    - Implement attention pattern extraction from REST API responses with layer-based 3D coordinates
    - Create activation flow data processing for token-level analysis with temporal 3D positioning
    - Write unit tests for 3D data extraction accuracy
    - _Requirements: 3.1, 3.2_

- [ ] 6. Build web interface integration
  - [x] 6.1 Create Visualization3DPanel component for NeuroScope
    - Write React/HTML component that embeds 3d-force-graph renderer with WebGL context
    - Implement 3D toolbar with layout, filter, camera controls, and 3D export controls
    - Create responsive 3D design that adapts to different screen sizes and maintains aspect ratio
    - Write component tests for 3D UI functionality (tested with Node.js test server)
    - _Requirements: 3.1, 3.3_

  - [ ] 6.2 Implement Visualization3DEmbedder for standalone 3D views
    - Write generateEmbedCode method creating self-contained 3D HTML with Three.js
    - Implement createStandaloneViewer for independent 3D graph applications
    - Create generateShareableLink for 3D graph sharing functionality with camera state
    - Add generate3DModelExport method for GLB/GLTF export
    - Write integration tests for 3D embedding functionality
    - _Requirements: 3.3, 5.1_

- [ ] 7. Implement layout and styling system
  - [ ] 7.1 Create Layout3DService with multiple 3D algorithms
    - Implement calculateCircuitLayout3D using 3D force-directed algorithm
    - Write calculateAttentionLayout3D with hierarchical 3D positioning and layer separation
    - Create calculateSphericalLayout3D for spherical 3D node arrangement
    - Create calculateComparisonLayout3D for side-by-side 3D visualization
    - Write unit tests for 3D layout algorithm consistency
    - _Requirements: 7.1, 7.2_

  - [ ] 7.2 Build 3D styling and theming system
    - Create 3D theme configurations for light, dark, and custom themes with proper lighting
    - Implement 3D color schemes and materials based on activation strength and component type
    - Write 3D node and link styling functions with Three.js materials and accessibility compliance
    - Create particle animation styling for link connections
    - Create unit tests for 3D styling consistency
    - _Requirements: 1.2, 2.2_

- [ ] 8. Add performance optimization features
  - [ ] 8.1 Implement 3D level-of-detail rendering for large graphs
    - Write 3D graph simplification algorithms for nodes above threshold with frustum culling
    - Create progressive 3D loading for graphs with 1000+ nodes using instanced rendering
    - Implement automatic 3D filtering based on activation strength and camera distance
    - Write performance tests validating smooth 60fps 3D interaction
    - _Requirements: 4.1, 4.2, 4.3_

  - [ ] 8.2 Create 3D caching and optimization system
    - Implement 3D graph data caching to reduce MLX Engine API calls
    - Write GPU memory management for large 3D visualization datasets
    - Create background processing for complex 3D layout calculations
    - Implement WebGL context optimization and Three.js performance tuning
    - Write performance benchmarks for 3D optimization validation
    - _Requirements: 4.1, 4.3_

- [ ] 9. Build export and sharing functionality
  - [ ] 9.1 Implement Export3DManager with multiple 3D format support
    - Write exportScreenshot method with configurable resolution, camera position, and 3D styling
    - Implement export3DModel method for GLB/GLTF format with materials and textures
    - Create exportJSON with complete 3D graph data, camera state, and metadata
    - Add exportVR method for VR-compatible formats where supported
    - Write unit tests for 3D export format accuracy
    - _Requirements: 5.1, 5.2, 5.3_

  - [ ] 9.2 Create interactive 3D export capabilities
    - Implement exportInteractive generating standalone HTML with 3d-force-graph and Three.js
    - Write 3D sharing functionality with URL-based graph state and camera position persistence
    - Create batch export for multiple 3D visualizations
    - Write integration tests for 3D sharing workflows
    - _Requirements: 5.1, 5.3_

- [ ] 10. Add comparison and analysis features
  - [ ] 10.1 Implement multi-3D-graph comparison system
    - Write side-by-side 3D visualization rendering with synchronized camera controls
    - Create 3D graph alignment tools for structural comparison in 3D space
    - Implement synchronized camera position, rotation, and zoom operations across 3D graphs
    - Write unit tests for 3D comparison functionality
    - _Requirements: 7.1, 7.2, 7.3_

  - [ ] 10.2 Build advanced 3D analysis tools
    - Create 3D node correspondence highlighting between comparison graphs with spatial effects
    - Implement 3D difference visualization showing structural changes in 3D space
    - Write similarity metrics for automated 3D graph comparison
    - Create integration tests for 3D analysis workflows
    - _Requirements: 7.3, 7.4_

- [ ] 11. Integrate with NeuroScope bridge
  - [ ] 11.1 Extend NeuroScope bridge for 3D graph data
    - Write Smalltalk conversion methods for 3d-force-graph data structures with 3D coordinates
    - Implement 3D graph data serialization compatible with NeuroScope format including camera state
    - Create bidirectional data flow between NeuroScope and 3D visualizations
    - Write integration tests for NeuroScope 3D compatibility
    - _Requirements: 3.1, 3.2_

  - [ ] 11.2 Create Smalltalk 3D visualization interface
    - Write Smalltalk methods for requesting 3D graph visualizations with camera parameters
    - Implement callback mechanisms for interactive 3D graph events (camera changes, node selection)
    - Create NeuroScope UI components for embedded 3D visualizations
    - Write end-to-end tests for NeuroScope 3D integration
    - _Requirements: 3.2, 3.3_

- [ ] 12. Add error handling and validation
  - [ ] 12.1 Implement comprehensive 3D error handling
    - Create Visualization3DError hierarchy for different 3D failure types (WebGL, Three.js, Camera)
    - Write graceful degradation for WebGL rendering failures with 2D fallback
    - Implement fallback 2D rendering for unsupported browsers or WebGL failures
    - Create unit tests for 3D error scenarios and recovery
    - _Requirements: 4.4_

  - [ ] 12.2 Build 3D data validation and sanitization
    - Write input validation for all 3D MCP tool parameters including camera coordinates
    - Implement 3D graph data validation before 3d-force-graph rendering
    - Create sanitization for user-provided 3D styling, materials, and camera configuration
    - Write security tests for 3D input validation
    - _Requirements: 6.1, 6.2, 6.3_

- [ ] 13. Create comprehensive testing suite
  - [ ] 13.1 Write unit tests for all core 3D components
    - Create tests for Graph3DConverter with various MLX data formats and 3D positioning
    - Write tests for ForceGraph3DRenderer with mock WebGL contexts and Three.js scenes
    - Implement tests for all 3D MCP tools with schema validation including camera parameters
    - Create performance tests for large 3D graph rendering and frame rate validation
    - _Requirements: 1.1, 2.1, 4.1_

  - [ ] 13.2 Build integration and end-to-end 3D tests
    - Write tests for complete MLX Engine to 3D visualization pipeline
    - Create tests for 3D MCP server tool integration
    - Implement browser WebGL compatibility tests across different environments
    - Write visual regression tests for consistent 3D rendering and camera positioning
    - _Requirements: 3.1, 8.1_

- [ ] 14. Update documentation and examples
  - [ ] 14.1 Update MCP server README with 3D visualization capabilities
    - Write comprehensive documentation for all 12 new 3D MCP tools including camera controls
    - Create usage examples for common 3D visualization workflows
    - Document 3D configuration options, WebGL requirements, and browser compatibility
    - Add troubleshooting guide for common 3D visualization issues (WebGL, Three.js, camera)
    - _Requirements: 8.1, 8.2, 8.3, 8.4, 8.5_

  - [ ] 14.2 Create 3D example applications and tutorials
    - Write example scripts demonstrating 3D circuit visualization workflows
    - Create tutorial for 3D attention pattern analysis using layer-separated visualizations
    - Implement example integration showing 3D MCP tool usage with camera controls
    - Write documentation for custom 3D visualization development with Three.js
    - Document Node.js test server setup for 3D development and WebGL testing
    - _Requirements: 6.1, 6.2, 6.3, 6.4_