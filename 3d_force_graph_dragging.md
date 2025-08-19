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

## Re-enabling Forces

If you want to turn forces back on later, you can restore them:

```javascript
// Re-enable the standard forces
graph.d3Force('charge', d3.forceManyBody().strength(-120))
     .d3Force('link', d3.forceLink().id(d => d.id).distance(100))
     .d3Force('center', d3.forceCenter(0, 0));

// Resume the animation
graph.resumeAnimation();

// Or restart the simulation entirely
graph.refresh();
```

### Custom Force Values

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
        // Re-enable forces
        graph.d3Force('charge', d3.forceManyBody().strength(-120))
             .d3Force('link', d3.forceLink().id(d => d.id).distance(100))
             .d3Force('center', d3.forceCenter(0, 0))
             .resumeAnimation();
        forcesEnabled = true;
    }
}
```

## Result

When you drag nodes with forces disabled, the connected links will stretch and deform to follow the dragged nodes, creating a "static" graph structure that can be manually manipulated. When forces are re-enabled, the graph will return to its dynamic, self-organizing behavior.

The library uses D3's force simulation under the hood, so you have full control over which forces are active and can create very flexible interaction models that can be toggled dynamically.