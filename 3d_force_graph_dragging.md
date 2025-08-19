# 3D Force Graph - Dragging Without Forces

Yes, the 3d-force-graph library can definitely create graphs with forces turned off while still allowing nodes to be dragged and deforming the graph structure.

## Disabling Forces

You can disable the simulation forces using:

```javascript
// Stop the force simulation
graph.d3Force('charge', null)
     .d3Force('link', null)
     .d3Force('center', null);

// Or pause the entire simulation
graph.pauseAnimation();
```

## Enabling Node Dragging

Even with forces disabled, you can still enable node dragging:

```javascript
// Enable node dragging (this is often enabled by default)
graph.enableNodeDrag(true);

// You can also add custom drag behavior
graph.onNodeDragEnd(node => {
    // Custom logic when drag ends
    console.log('Node dragged:', node);
});
```

## Example Setup

```javascript
const graph = ForceGraph3D()
    .graphData(myData)
    .enableNodeDrag(true)
    // Disable forces
    .d3Force('charge', null)
    .d3Force('link', null) 
    .d3Force('center', null)
    // Optional: set fixed positions
    .nodeThreeObject(node => {
        // You can set initial fixed positions
        node.fx = node.x;
        node.fy = node.y;
        node.fz = node.z;
        return someThreeObject;
    });
```

## Re-enabling Forces (Bulletproof Methods)

If you're working with an agent that can't seem to access the d3 instance, here are bulletproof ways to re-enable forces:

## Re-enabling Forces (Bulletproof Methods)

If you're working with an agent that can't seem to access the d3 instance, here are bulletproof ways to re-enable forces while preserving node positions:

### Method 1: Position-Preserving Graph Recreation
If you must recreate the graph, preserve all node positions first:

```javascript
// Capture current positions before recreation
const currentData = graph.graphData();
const positionMap = {};

currentData.nodes.forEach(node => {
    positionMap[node.id] = {
        x: node.x,
        y: node.y, 
        z: node.z,
        vx: node.vx || 0,
        vy: node.vy || 0,
        vz: node.vz || 0
    };
});

// Store original data with preserved positions
const preservedData = {
    nodes: currentData.nodes.map(node => ({
        ...node,
        fx: node.x, // Fix positions
        fy: node.y,
        fz: node.z
    })),
    links: currentData.links
};

// Recreate with preserved positions
const container = graph.graphEl();
graph = ForceGraph3D()(container)
    .graphData(preservedData)
    .enableNodeDrag(true)
    // Gradually unfix positions to allow forces
    .onEngineStop(() => {
        // After forces settle, unfix nodes
        setTimeout(() => {
            graph.graphData().nodes.forEach(node => {
                node.fx = null;
                node.fy = null;
                node.fz = null;
            });
        }, 1000);
    });
```

### Method 2: Force Simulation State Reset
Reset the simulation while preserving positions:

```javascript
// Get current node positions
const nodes = graph.graphData().nodes;
const links = graph.graphData().links;

// Store current positions
nodes.forEach(node => {
    node.fx = node.x;
    node.fy = node.y;
    node.fz = node.z;
});

// Force a complete simulation reset
try {
    const simulation = graph.d3Force();
    
    // Clear all forces
    simulation.force('charge', null)
              .force('link', null)
              .force('center', null)
              .stop();
    
    // Re-add forces with preserved positions
    simulation
        .force('charge', d3.forceManyBody().strength(-120))
        .force('link', d3.forceLink(links).id(d => d.id).distance(100))
        .force('center', d3.forceCenter(0, 0))
        .alpha(0.3) // Lower energy to preserve positions
        .restart();
        
    // Gradually release fixed positions
    setTimeout(() => {
        nodes.forEach(node => {
            node.fx = null;
            node.fy = null; 
            node.fz = null;
        });
        simulation.alpha(0.1).restart();
    }, 2000);
    
} catch (e) {
    console.error('Direct simulation access failed:', e);
}
```

### Method 3: Gradual Force Re-enabling
Enable forces gradually to minimize position disruption:

```javascript
function graduallyEnableForces(graph) {
    const nodes = graph.graphData().nodes;
    const links = graph.graphData().links;
    
    // Fix all current positions
    nodes.forEach(node => {
        node.fx = node.x;
        node.fy = node.y;
        node.fz = node.z;
    });
    
    try {
        // Start with very weak forces
        graph.d3Force('charge', d3.forceManyBody().strength(-5))
             .d3Force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(0.1))
             .d3Force('center', d3.forceCenter(0, 0).strength(0.1))
             .resumeAnimation();
        
        // Gradually increase force strength
        let strength = -5;
        let linkStrength = 0.1;
        
        const increaseForces = setInterval(() => {
            strength = Math.min(strength * 1.2, -30); // Target: -30
            linkStrength = Math.min(linkStrength * 1.1, 0.5); // Target: 0.5
            
            graph.d3Force('charge', d3.forceManyBody().strength(strength))
                 .d3Force('link', d3.forceLink(links).id(d => d.id).distance(100).strength(linkStrength));
            
            if (strength >= -30 && linkStrength >= 0.5) {
                clearInterval(increaseForces);
                
                // Finally release fixed positions
                setTimeout(() => {
                    nodes.forEach(node => {
                        node.fx = null;
                        node.fy = null;
                        node.fz = null;
                    });
                }, 1000);
            }
        }, 500);
        
    } catch (e) {
        console.error('Gradual force enabling failed:', e);
    }
}
```

### Method 4: Direct DOM/Canvas Manipulation Bypass
If all else fails, manipulate the underlying Three.js scene:

```javascript
function forceEnableBypassingGraph(graph) {
    try {
        // Access the Three.js scene directly
        const scene = graph.scene();
        const renderer = graph.renderer();
        
        // Store current object positions
        const positions = new Map();
        scene.traverse(obj => {
            if (obj.userData && obj.userData.nodeId) {
                positions.set(obj.userData.nodeId, {
                    x: obj.position.x,
                    y: obj.position.y,
                    z: obj.position.z
                });
            }
        });
        
        // Try to restart forces via multiple pathways
        const methods = [
            () => graph.refresh(),
            () => graph.d3Force().alpha(1).restart(),
            () => {
                const sim = graph.d3Force();
                sim.force('charge', d3.forceManyBody())
                   .force('link', d3.forceLink())
                   .force('center', d3.forceCenter())
                   .restart();
            }
        ];
        
        for (const method of methods) {
            try {
                method();
                console.log('Force restart successful');
                break;
            } catch (e) {
                console.warn('Method failed, trying next:', e);
            }
        }
        
        // Restore positions if they were disrupted
        setTimeout(() => {
            scene.traverse(obj => {
                if (obj.userData && obj.userData.nodeId && positions.has(obj.userData.nodeId)) {
                    const pos = positions.get(obj.userData.nodeId);
                    obj.position.set(pos.x, pos.y, pos.z);
                }
            });
        }, 100);
        
    } catch (e) {
        console.error('Direct manipulation failed:', e);
    }
}
```

### Method 5: Complete Position-Preserving Reset Function
Here's a bulletproof function that preserves positions no matter what:

```javascript
function forceEnableForces(graph, graphData) {
    // Always capture positions first
    const nodes = graphData?.nodes || graph.graphData().nodes;
    const links = graphData?.links || graph.graphData().links;
    
    const positionSnapshot = {};
    nodes.forEach(node => {
        positionSnapshot[node.id] = {
            x: node.x, y: node.y, z: node.z,
            vx: node.vx || 0, vy: node.vy || 0, vz: node.vz || 0
        };
    });
    
    const restorePositions = () => {
        nodes.forEach(node => {
            if (positionSnapshot[node.id]) {
                const pos = positionSnapshot[node.id];
                node.x = pos.x; node.y = pos.y; node.z = pos.z;
                node.vx = pos.vx; node.vy = pos.vy; node.vz = pos.vz;
            }
        });
    };
    
    try {
        // Method 1: Standard approach with position preservation
        nodes.forEach(node => { node.fx = node.x; node.fy = node.y; node.fz = node.z; });
        
        graph.d3Force('charge', d3.forceManyBody().strength(-120))
             .d3Force('link', d3.forceLink(links).id(d => d.id).distance(100))
             .d3Force('center', d3.forceCenter(0, 0))
             .resumeAnimation();
             
        // Gradually release positions
        setTimeout(() => {
            nodes.forEach(node => { node.fx = null; node.fy = null; node.fz = null; });
        }, 2000);
        
    } catch (e1) {
        restorePositions();
        try {
            // Method 2: Gradual enabling
            graduallyEnableForces(graph);
        } catch (e2) {
            restorePositions();
            try {
                // Method 3: Direct simulation access
                const sim = graph.d3Force();
                sim.force('charge', d3.forceManyBody())
                   .force('link', d3.forceLink(links))
                   .force('center', d3.forceCenter())
                   .alpha(0.1) // Very gentle restart
                   .restart();
            } catch (e3) {
                restorePositions();
                console.error('All force re-enabling methods failed');
                // Last resort: direct DOM manipulation
                forceEnableBypassingGraph(graph);
            }
        }
    }
    
    return graph;
}
```

### Method 2: Use refresh() to Reset Everything
This forces a complete restart of the simulation:

```javascript
// Clear any disabled forces and restart
graph.d3Force('charge', d3.forceManyBody())
     .d3Force('link', d3.forceLink())
     .d3Force('center', d3.forceCenter())
     .refresh(); // This completely restarts the simulation
```

### Method 3: Access D3 Simulation Directly
If the graph instance methods aren't working, access the underlying D3 simulation:

```javascript
// Get the underlying D3 simulation
const simulation = graph.d3Force();

// Restart with default forces
simulation
    .force('charge', d3.forceManyBody().strength(-30))
    .force('link', d3.forceLink(originalData.links).id(d => d.id))
    .force('center', d3.forceCenter(0, 0))
    .alpha(1) // Reset simulation energy
    .restart(); // Restart the simulation
```

### Method 4: Complete Reset Function
Here's a bulletproof function that tries multiple approaches:

```javascript
function forceEnableForces(graph, graphData) {
    try {
        // Method 1: Try standard approach
        graph.d3Force('charge', d3.forceManyBody().strength(-120))
             .d3Force('link', d3.forceLink().id(d => d.id).distance(100))
             .d3Force('center', d3.forceCenter(0, 0))
             .resumeAnimation();
    } catch (e1) {
        try {
            // Method 2: Try refresh
            graph.refresh();
        } catch (e2) {
            try {
                // Method 3: Access simulation directly
                const sim = graph.d3Force();
                sim.force('charge', d3.forceManyBody())
                   .force('link', d3.forceLink(graphData.links))
                   .force('center', d3.forceCenter())
                   .alpha(1)
                   .restart();
            } catch (e3) {
                // Method 4: Position-preserving recreation (last resort)
                console.warn('Using position-preserving recreation');
                graph = recreateWithPositions(graph, graphData);
            }
        }
    }
    return graph;
}
```

### Method 5: Window/Global Scope Access
If the agent has limited scope, ensure d3 is globally accessible:

```javascript
// Make sure d3 is available globally
window.d3 = d3;

// Use fully qualified references
graph.d3Force('charge', window.d3.forceManyBody().strength(-120))
     .d3Force('link', window.d3.forceLink().id(d => d.id))
     .d3Force('center', window.d3.forceCenter(0, 0));
```

## Custom Force Values

You can also customize the force strengths when re-enabling:

```javascript
// Weaker repulsion force
graph.d3Force('charge', d3.forceManyBody().strength(-50));

// Longer link distance
graph.d3Force('link', d3.forceLink().id(d => d.id).distance(200));

// Different center point
graph.d3Force('center', d3.forceCenter(100, 50));
```

### Toggle Function Example

Here's a complete example of toggling forces on/off:

```javascript
let forcesEnabled = true;

function toggleForces() {
    if (forcesEnabled) {
        // Disable forces
        graph.d3Force('charge', null)
             .d3Force('link', null)
             .d3Force('center', null)
             .pauseAnimation();
        forcesEnabled = false;
    } else {
        // Re-enable forces using bulletproof method
        graph = forceEnableForces(graph, graph.graphData());
        forcesEnabled = true;
    }
}
```

## Result

When you drag nodes with forces disabled, the connected links will stretch and deform to follow the dragged nodes, creating a "static" graph structure that can be manually manipulated. When forces are re-enabled, the graph will return to its dynamic, self-organizing behavior.

The library uses D3's force simulation under the hood, so you have full control over which forces are active and can create very flexible interaction models that can be toggled dynamically.