/**
 * Core mechanistic interpretability tools
 * Basic implementation for testing the MCP framework
 */

import { getLogger } from '../utils/logging.js';

const logger = getLogger('CoreTools');

/**
 * Example core_discover_circuits tool
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Circuit discovery results
 */
async function coreDiscoverCircuits(params) {
  logger.info(`Circuit discovery requested for phenomenon: ${params.phenomenon}`);

  // AGENT.md: Never fake anything. Report exact parameters and error details.
  return {
    success: false,
    error: "Circuit discovery not yet implemented in MLX Engine. No mock data per AGENT.md guidelines.",
    phenomenon: params.phenomenon,
    model_id: params.model_id,
    exact_parameters_passed: params,
    note: "Real circuit discovery requires implementing activation patching and causal tracing in the MLX Engine API"
  };
}

/**
 * Example core_localize_features tool
 * @param {object} params - Tool parameters
 * @returns {Promise<object>} Feature localization results
 */
async function coreLocalizeFeatures(params) {
  logger.info(`Feature localization requested: ${params.feature_name}`);

  // AGENT.md: Never fake anything. Report exact parameters and error details.
  return {
    success: false,
    error: "Feature localization not yet implemented in MLX Engine. No mock data per AGENT.md guidelines.",
    feature_name: params.feature_name,
    exact_parameters_passed: params,
    note: "Real feature localization requires implementing PCA and probing classifiers in the MLX Engine API"
  };
}

/**
 * Example ping tool for testing
 * @param {object} params - Tool parameters
 * @returns {object} Ping response
 */
function ping(params) {
  return {
    success: true,
    message: 'pong',
    timestamp: new Date().toISOString(),
    echo: params,
  };
}

// Tool definitions with comprehensive JSON schemas
export const coreTools = [
  {
    name: 'core_discover_circuits',
    description:
      'Discovers circuits for a specific phenomenon using causal tracing with activation patching',
    inputSchema: {
      type: 'object',
      properties: {
        phenomenon: {
          type: 'string',
          description: 'The target phenomenon to find circuits for',
          enum: ['IOI', 'indirect_object_identification', 'arithmetic', 'factual_recall'],
        },
        model_id: {
          type: 'string',
          description: 'Identifier of the loaded model to analyze',
        },
        confidence_threshold: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.7,
          description: 'Minimum confidence score for circuit candidates',
        },
        max_circuits: {
          type: 'integer',
          minimum: 1,
          maximum: 50,
          default: 10,
          description: 'Maximum number of circuit candidates to return',
        },
      },
      required: ['phenomenon', 'model_id'],
      additionalProperties: false,
    },
    outputSchema: {
      type: 'object',
      properties: {
        success: { type: 'boolean' },
        circuits: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              id: { type: 'string' },
              name: { type: 'string' },
              confidence: { type: 'number', minimum: 0, maximum: 1 },
              layers: { type: 'array', items: { type: 'integer' } },
              components: { type: 'array', items: { type: 'string' } },
              validation_metrics: {
                type: 'object',
                properties: {
                  performance_recovery: { type: 'number' },
                  attribution_score: { type: 'number' },
                  consistency_score: { type: 'number' },
                },
                required: ['performance_recovery', 'attribution_score', 'consistency_score'],
              },
            },
            required: ['id', 'name', 'confidence', 'layers', 'components', 'validation_metrics'],
          },
        },
        execution_time_ms: { type: 'number' },
        model_info: {
          type: 'object',
          properties: {
            model_id: { type: 'string' },
            architecture: { type: 'string' },
            num_layers: { type: 'integer' },
          },
          required: ['model_id', 'architecture', 'num_layers'],
        },
      },
      required: ['success', 'circuits', 'execution_time_ms', 'model_info'],
      additionalProperties: false,
    },
    handler: coreDiscoverCircuits,
  },
  {
    name: 'core_localize_features',
    description:
      'Localizes neurons responsible for specific features using PCA and probing classifiers',
    inputSchema: {
      type: 'object',
      properties: {
        feature_name: {
          type: 'string',
          description: 'Name of the feature to localize',
        },
        model_id: {
          type: 'string',
          description: 'Identifier of the loaded model to analyze',
        },
        layer_range: {
          type: 'object',
          properties: {
            start: { type: 'integer', minimum: 0 },
            end: { type: 'integer', minimum: 0 },
          },
          description: 'Range of layers to analyze',
        },
        threshold: {
          type: 'number',
          minimum: 0,
          maximum: 1,
          default: 0.8,
          description: 'Activation threshold for feature detection',
        },
      },
      required: ['feature_name', 'model_id'],
      additionalProperties: false,
    },
    outputSchema: {
      type: 'object',
      properties: {
        success: { type: 'boolean' },
        feature_name: { type: 'string' },
        neurons: {
          type: 'array',
          items: {
            type: 'object',
            properties: {
              layer: { type: 'integer' },
              neuron_id: { type: 'integer' },
              activation_strength: { type: 'number' },
              confidence: { type: 'number' },
            },
            required: ['layer', 'neuron_id', 'activation_strength', 'confidence'],
          },
        },
        validation_metrics: {
          type: 'object',
          properties: {
            valid: { type: 'boolean' },
            confidence: { type: 'number' },
            metrics: { type: 'object' },
          },
          required: ['valid', 'confidence', 'metrics'],
        },
      },
      required: ['success', 'feature_name', 'neurons', 'validation_metrics'],
      additionalProperties: false,
    },
    handler: coreLocalizeFeatures,
  },
  {
    name: 'ping',
    description: 'Simple ping tool for testing server connectivity',
    inputSchema: {
      type: 'object',
      properties: {
        message: {
          type: 'string',
          description: 'Optional message to echo back',
        },
      },
      additionalProperties: false,
    },
    outputSchema: {
      type: 'object',
      properties: {
        success: { type: 'boolean' },
        message: { type: 'string' },
        timestamp: { type: 'string' },
        echo: { type: 'object' },
      },
      required: ['success', 'message', 'timestamp'],
      additionalProperties: false,
    },
    handler: ping,
  },
];
