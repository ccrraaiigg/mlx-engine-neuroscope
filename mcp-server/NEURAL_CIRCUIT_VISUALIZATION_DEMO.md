# Neural Circuit Visualization Demo Guide

This guide documents how to create real-time, interactive
visualizations of neural circuits from GPT-OSS-20B using the
mechanistic interpretability MCP server.

## 🚨 CRITICAL: MCP SERVER RESTART REQUIREMENT 🚨

**MANDATORY FOR ALL CODE CHANGES**: Every time you modify ANY MCP server code, the MCP server MUST be restarted by the user before testing changes.

### ⛔ ABSOLUTE PROHIBITION ⛔
**NEVER MODIFY MCP SERVER CODE WITHOUT USER RESTART**
- Modifying MCP server code and then calling MCP tools is **STRICTLY FORBIDDEN**
- This violates the code modification policy and wastes computational resources
- **ALL CODE CHANGES ARE INEFFECTIVE** until the server is manually restarted
- Running tools after code changes tests **OLD CODE**, not your modifications

### Why This Is Critical:
- **Code changes don't take effect** until the server is restarted
- **Testing old code** leads to debugging non-existent issues
- **Model state is reset** on restart, requiring full demo restart
- **Generated data is deleted** on restart
- **Violating this policy wastes resources** and creates confusion

### Required Restart Protocol:
1. **Make code changes** to MCP server files
2. **Increment version number** in the version tool
3. **STOP IMMEDIATELY** - Do not call any MCP tools
4. **Ask user to restart** the MCP server in IDE
5. **Wait for user confirmation** of restart
6. **Verify restart** by calling `version` tool first
7. **Confirm version match** before proceeding
8. **Restart demo from beginning** (model loading + full workflow)

**⚠️ NEVER assume code changes work without restart verification!**
**⚠️ NEVER call MCP tools immediately after modifying MCP server code!**

---

You are forbidden to run 'node' or 'python3' or 'python'. The only way
you can run code is via the MCP server.

DO NOT EDIT real_circuit_data.json or real_circuit.html.
DO NOT EDIT real_circuit_data.json or real_circuit.html.
DO NOT EDIT real_circuit_data.json or real_circuit.html.
DO NOT EDIT real_circuit_data.json or real_circuit.html.
DO NOT EDIT real_circuit_data.json or real_circuit.html.

Obey AGENT.md at all times. Never make up data, or modify data that
comes from an MCP server tool; that includes json and html files. All
data must come from a running model, via the MCP server tools. If
there's a problem with a visualization, you must fix the MCP server
code. Every time you change the MCP server code, it must be reloaded
manually by the user. You cannot test a change to the MCP server code
without the user reloading the MCP server first. When the MCP server
is reloaded, the model is killed and all generated data is deleted;
you have to run the demo again from the beginning.

## Critical Development Workflow

### ⚠️ POLICY VIOLATION WARNING ⚠️

**MODIFYING MCP SERVER CODE AND IMMEDIATELY CALLING MCP TOOLS IS A SERIOUS VIOLATION**

This practice:
- **Wastes computational resources** by testing old, unchanged code
- **Creates false debugging scenarios** where you fix non-existent problems
- **Violates the fundamental principle** that code changes require server restart
- **Demonstrates poor understanding** of the MCP server architecture

**If you modify MCP server code, you MUST ask the user to restart before any testing.**

### 🔄 MANDATORY CODE MODIFICATION PROTOCOL 🔄

**STOP! READ THIS BEFORE ANY CODE CHANGES!**

Every single time you modify ANY MCP server code file, you MUST follow this exact protocol:

#### Step 1: Before Making Changes
- Note the current version number from last `version` tool call
- Plan your changes carefully to minimize restart cycles

#### Step 2: Make Code Changes
- Edit the MCP server files as needed
- **IMMEDIATELY increment version number** in the version tool
- Document what you changed for verification

#### Step 3: Request User Restart (CRITICAL)
- **STOP ALL TESTING** - code changes are not active yet
- **Ask user explicitly**: "Please restart the MCP server in your IDE for the code changes to take effect"
- **Wait for user confirmation** before proceeding

#### Step 4: Verify Restart Worked
- **First tool call MUST be `version`** to verify restart
- **Check version number matches** your increment
- If version doesn't match: **Ask user to restart again**

#### Step 5: Restart Demo Workflow
- **Model is killed** on restart - reload it
- **All data is deleted** - restart from health check
- **Run full demo sequence** from beginning

### 🚫 COMMON MISTAKES TO AVOID:
- ❌ Testing code changes without restart verification
- ❌ Assuming changes work because "they should"
- ❌ Debugging issues that don't exist in the new code
- ❌ Continuing with old model/data after restart

### 🎯 SUCCESS INDICATORS:
- ✅ Version number matches your increment
- ✅ User confirmed they restarted the server
- ✅ Fresh model loaded successfully
- ✅ Demo runs with new code behavior

For your own debugging, use the IDE's webpage preview tool, not the
MCP server open_browser tool. That way you can see console errors
yourself.

**Why This Matters**: The version number is the only reliable way to confirm that code changes have been loaded by the MCP server. Without this verification, you may be testing old code and debugging non-existent issues.

## Required Flow for Circuit Discovery and Visualization

**IMPORTANT**: The following sequence of MCP tools must be executed in order to properly discover and visualize neural circuits:

### 1. Version Check

- Verifies the MCP server version and recent changes
- Ensures compatibility with the latest features

### 2. Health Check

- Confirms that both MLX Engine and visualization servers are running
- Verifies model availability and server connectivity

### 3. Load Model

- Ensures a fresh initial state for the model.

### 4. Discover Circuits

- Identifies circuits for specific phenomena (IOI, arithmetic, factual_recall)
- Uses causal tracing with activation patching
- Returns circuit data with layer information, components, and confidence scores

### 5. Circuit Diagram Generation

- Creates interactive 3D visualization using the discovered circuit data
- Generates both JSON data file and HTML visualization

### 6. Open Visualization
- Open the generated HTML file in browser to view the interactive circuit diagram
- The visualization shows nodes (neural components) and edges (connections)
- Supports 3D navigation and interaction

**Note**: All data must come from live model analysis via MCP tools - no synthetic or hardcoded data should be used.

## Overview

We have successfully built a complete end-to-end pipeline that:
1. **Discovers neural circuits** automatically using real activation capture and causal analysis
2. **Localizes specific features** across multiple transformer layers using activation comparison
3. **Captures real activation data** from GPT-OSS-20B during inference
4. **Converts raw activations** to interactive graph format (nodes and links)
5. **Renders beautiful WebGL2 visualizations** using 3D Force Graph
6. **Provides interactive exploration** with node hovering and real-time updates

## System Architecture

```
GPT-OSS-20B Model → MLX Engine → MCP Server → 3D Force Graph WebGL2 → Browser
     (Inference)    (Capture)   (Convert)      (Visualize)        (Interact)
```

## Key Components

### 1. MLX Engine (Port 50111)
- Runs GPT-OSS-20B model with activation hooks
- Captures neural activations during text generation
- Provides REST API endpoints for model interaction

### 2. MCP Server
- Mechanistic Interpretability MCP Server (Version 107+)
- 16+ tools for neural analysis and visualization
- **Real circuit discovery** using activation patching and causal tracing
- **Real feature localization** using activation comparison across layers
- Real-time data processing and conversion

### 3. 3D Force Graph Visualization
- Professional WebGL2-based graph renderer
- Hardware-accelerated interactive visualizations
- Node hovering, clicking, and real-time updates

## Demo Instructions

**Important**: This demo is driven by **conversation with the user**. The agent should respond to user requests by invoking the appropriate MCP tools. The following steps represent a typical demo flow, but the agent should adapt based on user questions and interests.

### Typical Demo Flow

When a user asks to see the neural circuit visualization demo, the agent should:

1. **Verify system status** by checking MCP server version and service health
2. **Load model** if not already loaded using the real model loading tool
3. **Discover circuits** automatically for specific phenomena (arithmetic, factual recall, etc.)
4. **Create interactive visualization** from the captured data
5. **Open browser** to display the WebGL2 visualization
6. **Explain the results** and answer user questions about neural circuits discovered

### Step 1: Verify System Status (Agent Action)

The agent should first check system status when requested:

#### Check MCP Server Version

**Important**: The version number indicates the server state and must be incremented every time the MCP server code is modified.

#### Check All Services Health

#### Start Services (if needed)
If any service is unreachable, the agent should start it.

### Step 2: Load Model (Agent Action)

**CRITICAL: Always reload the model for clean analysis!**

Even if a model is already loaded, the agent should reload it to ensure clean state:

**Why reload?** Previous analyses may have influenced the model's internal state (attention patterns, residual stream, activation baselines). For accurate mechanistic interpretability research, each visualization should analyze the model's **natural processing** of each math problem, not processing after being "primed" with previous mathematical operations.

### Step 3: Discover Neural Circuits (Agent Action)

The agent can now discover real neural circuits for specific phenomena.

### Step 4: Create Interactive Visualization (Agent Action)

The agent automatically uses the captured data to create the visualization.

### Step 5: Open Browser Visualization (Agent Action)

The agent opens the visualization in the user's browser.

**User Interaction**: The agent should explain: *"I've opened the interactive visualization in your browser. You can explore how the neural network processes the math problem."*

## Expected Results

### What the User Will See

The agent should explain what's happening as the demo progresses and help the user understand the visualization.

### Interactive Features
- **Informative Node Labels**
- **Color Coding**: Red for MLP components, Blue for attention
- **Real Metadata**: Actual tensor shapes and component counts
- **Live Updates**: Node/link counts update in real-time

### Visual Elements
- **3 Nodes**: Representing the captured neural components
- **2 Links**: Showing activation flow between layers
- **WebGL2 Rendering**: Hardware-accelerated smooth graphics
- **Responsive Layout**: Adapts to browser window size

## Technical Details

### Data Flow
1. **Raw Activations**: Captured from specific model layers during inference
2. **Server-side Conversion**: Transforms activations into `{nodes: [], links: []}` format
3. **Browser-side Rendering**: 3D Force Graph processes data for WebGL2 display
4. **Interactive Updates**: Real-time hover and click event handling

## Critical Best Practices for Clean Analysis

### Model State Management

**Always reload the model between visualization demos:**

**Why this matters:**
- **Clean slate**: Each analysis starts with pristine model state
- **No contamination**: Previous mathematical operations don't influence current analysis
- **Research accuracy**: Captures natural processing, not primed responses
- **Reproducibility**: Consistent baseline for each demonstration

### Feature Localization (NEW - Version 107+)

The MCP server provides **real feature localization** across transformer layers:

**Supported Features:**
- `"sentiment"` - Emotion processing localization
- `"syntax"` - Grammar and structure processing
- `"factual"` - Factual knowledge representation
- `"arithmetic"` - Mathematical reasoning localization
- `"negation"` - Negation handling mechanisms

**Real Analysis:**
- Multi-layer activation comparison
- Genuine localization strength scores
- Real prompt testing with agent-provided contrasting examples
- Authentic layer-by-layer analysis

### Model Loading (Real Implementation)

**Real Model Management:**
- Loads actual GPT-OSS-20B model from LM Studio path
- Genuine activation hook support verification
- Real model metadata (number of layers and hidden size)
- Authentic model status reporting

## Demo Variations

The agent should adapt to user interests and questions:

### Different Problem Types

### Interactive Exploration  
The agent can respond to user questions like:
- *"What do the red vs blue nodes represent?"*
- *"How does the network solve multiplication differently than addition?"*
- *"Can we see deeper layers of the network?"*
- *"What happens with a more complex problem?"*

### Layer Exploration
- Modify capture hooks to explore different layers (0-19)
- Compare attention vs MLP activation patterns
- Study information flow across transformer layers

### Advanced Features
For users wanting deeper analysis:
- Real-time activation streaming
- Multi-layer circuit discovery
- Attention pattern visualization
- MLP decomposition analysis

## Next Steps

1. **Circuit Discovery**: Automated identification of computational circuits.
2. **Feature Localization**: Multi-layer feature analysis
3. **Model Loading**: Real model management
4. **Enhanced Visualization**: Integrate circuit discovery results into 3D graphs
5. **Attention Patterns**: Visualize attention weight matrices from discovered circuits
6. **Real-time Streaming**: Live activation updates during generation
7. **Comparative Analysis**: Side-by-side circuit comparisons across phenomena
8. **Circuit Editing**: Modify discovered circuits and observe behavioral changes

---

*This system represents a complete pipeline for mechanistic interpretability research, enabling real-time exploration of neural network internals with professional-grade visualization tools.*

