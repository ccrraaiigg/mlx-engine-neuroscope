/**
 * MLX Engine Integration Tools
 * 
 * Tools that integrate with the MLX Engine REST API to perform
 * real model analysis and activation capture.
 */

import { MLXEngineClient } from './mlx_engine_client.js';
import { getLogger } from '../utils/logging.js';

const logger = getLogger('MLXTools');

// Global MLX client instance
let mlxClient = null;

/**
 * Initializes the MLX Engine client
 * @param {object} config - MLX Engine configuration
 */
export function initializeMLXClient(config) {
  mlxClient = new MLXEngineClient(config);
  logger.info(`Initialized MLX Engine client for ${config.apiUrl}`);
}

/**
 * Gets the current MLX client instance
 * @returns {MLXEngineClient} MLX client
 */
export function getMLXClient() {
  if (!mlxClient) {
    throw new Error('MLX Engine client not initialized. Call initializeMLXClient() first.');
  }
  return mlxClient;
}

/**
 * Loads a model in MLX Engine
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Load result
 */
async function mlxLoadModel(params) {
  logger.info(`Loading model: ${params.model_id}`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.loadModel(params.model_id, {
      quantization: params.quantization,
      max_context_length: params.max_context_length,
      device: params.device,
    });
    
    return {
      success: true,
      model_id: params.model_id,
      status: result.status,
      parameters: result.parameters,
      quantization: result.quantization,
      max_context_length: result.max_context_length,
      load_time_ms: result.load_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to load model: ${error.message}`);
    return {
      success: false,
      error: error.message,
      model_id: params.model_id,
    };
  }
}

/**
 * Creates activation hooks for capturing model internals
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Hook creation result
 */
async function mlxCreateHooks(params) {
  logger.info(`Creating hooks for layers: ${params.layers.join(', ')}`);
  
  const client = getMLXClient();
  
  const hookSpecs = params.layers.map(layer => ({
    layer: layer,
    hook_type: params.hook_type || 'activation',
    components: params.components || ['mlp', 'attention'],
    capture_gradients: params.capture_gradients || false,
  }));
  
  try {
    const result = await client.createHooks(hookSpecs);
    
    return {
      success: true,
      hooks_created: result.hooks_created,
      hook_ids: result.hook_ids,
      layers: params.layers,
      components: params.components,
    };
    
  } catch (error) {
    logger.error(`Failed to create hooks: ${error.message}`);
    return {
      success: false,
      error: error.message,
      layers: params.layers,
    };
  }
}

/**
 * Captures activations during text generation
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Generation with activations
 */
async function mlxCaptureActivations(params) {
  logger.info(`Capturing activations for prompt: "${params.prompt.substring(0, 50)}..."`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.generateWithActivations(params.prompt, {
      max_tokens: params.max_tokens,
      temperature: params.temperature,
      top_p: params.top_p,
      capture_attention: params.capture_attention,
      capture_residual_stream: params.capture_residual_stream,
    });
    
    return {
      success: true,
      prompt: params.prompt,
      generated_text: result.text,
      activations: result.activations,
      attention: result.attention,
      residual_stream: result.residual_stream,
      generation_time_ms: result.generation_time_ms,
      tokens_generated: result.tokens_generated,
    };
    
  } catch (error) {
    logger.error(`Failed to capture activations: ${error.message}`);
    return {
      success: false,
      error: error.message,
      prompt: params.prompt,
    };
  }
}

/**
 * Analyzes mathematical reasoning circuits
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Math analysis results
 */
async function mlxAnalyzeMath(params) {
  logger.info(`Analyzing mathematical reasoning: "${params.prompt}"`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.analyzeMath(params.prompt, {
      max_tokens: params.max_tokens,
      temperature: params.temperature,
      analysis_depth: params.analysis_depth,
    });
    
    return {
      success: true,
      prompt: params.prompt,
      analysis_type: 'mathematical_reasoning',
      circuits: result.circuits,
      operations_detected: result.operations_detected,
      confidence_scores: result.confidence_scores,
      execution_time_ms: result.execution_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to analyze math: ${error.message}`);
    return {
      success: false,
      error: error.message,
      prompt: params.prompt,
    };
  }
}

/**
 * Analyzes attention patterns
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Attention analysis results
 */
async function mlxAnalyzeAttention(params) {
  logger.info(`Analyzing attention patterns for layers: ${params.layers.join(', ')}`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.analyzeAttention(params.prompt, params.layers);
    
    return {
      success: true,
      prompt: params.prompt,
      layers: params.layers,
      analysis_type: 'attention_patterns',
      patterns: result.patterns,
      attention_heads: result.attention_heads,
      pattern_types: result.pattern_types,
      execution_time_ms: result.execution_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to analyze attention: ${error.message}`);
    return {
      success: false,
      error: error.message,
      prompt: params.prompt,
      layers: params.layers,
    };
  }
}

/**
 * Analyzes factual recall circuits
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Factual analysis results
 */
async function mlxAnalyzeFactual(params) {
  logger.info(`Analyzing factual recall: "${params.query}"`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.analyzeFactual(params.query, {
      max_tokens: params.max_tokens,
      analysis_depth: params.analysis_depth,
    });
    
    return {
      success: true,
      query: params.query,
      analysis_type: 'factual_recall',
      facts_detected: result.facts_detected,
      retrieval_mechanisms: result.retrieval_mechanisms,
      circuit_components: result.circuit_components,
      confidence_scores: result.confidence_scores,
      execution_time_ms: result.execution_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to analyze factual recall: ${error.message}`);
    return {
      success: false,
      error: error.message,
      query: params.query,
    };
  }
}

/**
 * Tracks residual stream flow
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Residual stream analysis
 */
async function mlxTrackResidual(params) {
  logger.info(`Tracking residual stream: "${params.prompt}"`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.trackResidualStream(params.prompt, {
      layers: params.layers,
      components: params.components,
    });
    
    return {
      success: true,
      prompt: params.prompt,
      analysis_type: 'residual_flow',
      flow_data: result.flow_data,
      layer_contributions: result.layer_contributions,
      information_flow: result.information_flow,
      execution_time_ms: result.execution_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to track residual stream: ${error.message}`);
    return {
      success: false,
      error: error.message,
      prompt: params.prompt,
    };
  }
}

/**
 * Exports analysis data to NeuroScope format
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Export result
 */
async function mlxExportNeuroScope(params) {
  logger.info(`Exporting to NeuroScope format: ${params.format}`);
  
  const client = getMLXClient();
  
  try {
    const result = await client.exportNeuroScope(params.analysis_data, params.format);
    
    return {
      success: true,
      format: params.format,
      export_data: result.export_data,
      smalltalk_code: result.smalltalk_code,
      visualization_config: result.visualization_config,
      export_time_ms: result.export_time_ms,
    };
    
  } catch (error) {
    logger.error(`Failed to export to NeuroScope: ${error.message}`);
    return {
      success: false,
      error: error.message,
      format: params.format,
    };
  }
}

// Tool definitions with comprehensive schemas
export const mlxTools = [
  {
    name: 'mlx_load_model',
    description: 'Loads a model in the MLX Engine for analysis',
    inputSchema: {
      type: 'object',
      properties: {
        model_id: {
          type: 'string',
          description: 'Model identifier (e.g., gpt-oss-20b, llama-2-7b)',
          enum: ['gpt-oss-20b', 'llama-2-7b', 'mistral-7b', 'phi-2'],
        },
        quantization: {
          type: 'string',
          description: 'Quantization level',
          enum: ['none', '4bit', '8bit'],
          default: 'none',
        },
        max_context_length: {
          type: 'integer',
          minimum: 512,
          maximum: 131072,
          default: 2048,
          description: 'Maximum context length',
        },
        device: {
          type: 'string',
          description: 'Device to load model on',
          enum: ['auto', 'cpu', 'mps', 'cuda'],
          default: 'auto',
        },
      },
      required: ['model_id'],
      additionalProperties: false,
    },
    handler: mlxLoadModel,
  },
  
  {
    name: 'mlx_create_hooks',
    description: 'Creates activation hooks for capturing model internals',
    inputSchema: {
      type: 'object',
      properties: {
        layers: {
          type: 'array',
          items: { type: 'integer', minimum: 0, maximum: 50 },
          description: 'Layer indices to hook',
          minItems: 1,
        },
        hook_type: {
          type: 'string',
          enum: ['activation', 'gradient', 'both'],
          default: 'activation',
          description: 'Type of data to capture',
        },
        components: {
          type: 'array',
          items: { type: 'string', enum: ['mlp', 'attention', 'residual', 'all'] },
          default: ['mlp', 'attention'],
          description: 'Model components to hook',
        },
        capture_gradients: {
          type: 'boolean',
          default: false,
          description: 'Whether to capture gradients',
        },
      },
      required: ['layers'],
      additionalProperties: false,
    },
    handler: mlxCreateHooks,
  },
  
  {
    name: 'mlx_capture_activations',
    description: 'Captures activations during text generation',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'Input prompt for generation',
          minLength: 1,
          maxLength: 10000,
        },
        max_tokens: {
          type: 'integer',
          minimum: 1,
          maximum: 1000,
          default: 100,
          description: 'Maximum tokens to generate',
        },
        temperature: {
          type: 'number',
          minimum: 0,
          maximum: 2,
          default: 0.7,
          description: 'Generation temperature',
        },
        top_p: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.9,
          description: 'Top-p sampling parameter',
        },
        capture_attention: {
          type: 'boolean',
          default: true,
          description: 'Whether to capture attention patterns',
        },
        capture_residual_stream: {
          type: 'boolean',
          default: false,
          description: 'Whether to capture residual stream',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    handler: mlxCaptureActivations,
  },
  
  {
    name: 'mlx_analyze_math',
    description: 'Analyzes mathematical reasoning circuits in the model',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'Mathematical problem or expression',
          minLength: 1,
        },
        max_tokens: {
          type: 'integer',
          minimum: 10,
          maximum: 500,
          default: 100,
          description: 'Maximum tokens for analysis',
        },
        temperature: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.1,
          description: 'Low temperature for consistent math',
        },
        analysis_depth: {
          type: 'string',
          enum: ['shallow', 'medium', 'deep'],
          default: 'medium',
          description: 'Depth of circuit analysis',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    handler: mlxAnalyzeMath,
  },
  
  {
    name: 'mlx_analyze_attention',
    description: 'Analyzes attention patterns in specified layers',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'Input text to analyze attention for',
          minLength: 1,
        },
        layers: {
          type: 'array',
          items: { type: 'integer', minimum: 0, maximum: 50 },
          description: 'Layers to analyze attention in',
          minItems: 1,
          maxItems: 10,
        },
      },
      required: ['prompt', 'layers'],
      additionalProperties: false,
    },
    handler: mlxAnalyzeAttention,
  },
  
  {
    name: 'mlx_analyze_factual',
    description: 'Analyzes factual recall circuits and mechanisms',
    inputSchema: {
      type: 'object',
      properties: {
        query: {
          type: 'string',
          description: 'Factual query to analyze',
          minLength: 1,
        },
        max_tokens: {
          type: 'integer',
          minimum: 5,
          maximum: 200,
          default: 50,
          description: 'Maximum tokens for factual response',
        },
        analysis_depth: {
          type: 'string',
          enum: ['shallow', 'medium', 'deep'],
          default: 'medium',
          description: 'Depth of factual analysis',
        },
      },
      required: ['query'],
      additionalProperties: false,
    },
    handler: mlxAnalyzeFactual,
  },
  
  {
    name: 'mlx_track_residual',
    description: 'Tracks information flow through the residual stream',
    inputSchema: {
      type: 'object',
      properties: {
        prompt: {
          type: 'string',
          description: 'Input text to track residual flow for',
          minLength: 1,
        },
        layers: {
          type: 'array',
          items: { type: 'integer', minimum: 0, maximum: 50 },
          description: 'Layers to track (empty for all)',
          default: [],
        },
        components: {
          type: 'array',
          items: { type: 'string', enum: ['attention', 'mlp', 'both'] },
          default: ['both'],
          description: 'Components to track contributions from',
        },
      },
      required: ['prompt'],
      additionalProperties: false,
    },
    handler: mlxTrackResidual,
  },
  
  {
    name: 'mlx_export_neuroscope',
    description: 'Exports analysis data to NeuroScope format',
    inputSchema: {
      type: 'object',
      properties: {
        analysis_data: {
          type: 'object',
          description: 'Analysis data to export',
        },
        format: {
          type: 'string',
          enum: ['smalltalk', 'json', 'both'],
          default: 'smalltalk',
          description: 'Export format',
        },
      },
      required: ['analysis_data'],
      additionalProperties: false,
    },
    handler: mlxExportNeuroScope,
  },
];