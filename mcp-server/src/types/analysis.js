/**
 * Analysis types and schemas
 */

import { z } from 'zod';

// Analysis result status enum
export const ResultStatusSchema = z.enum(['success', 'error', 'partial', 'timeout']);

// Validation result schema
export const ValidationResultSchema = z.object({
  valid: z.boolean(),
  confidence: z.number().min(0).max(1),
  metrics: z.record(z.number()),
  errors: z.array(z.string()).optional(),
});

// Analysis result schema
export const AnalysisResultSchema = z.object({
  operation: z.string(),
  status: ResultStatusSchema,
  data: z.record(z.any()),
  confidence: z.number().min(0).max(1).optional(),
  validation: ValidationResultSchema.optional(),
  timestamp: z.string(),
  execution_time: z.number(),
});

// Activation data schema
export const ActivationDataSchema = z.object({
  model_id: z.string(),
  layer_activations: z.record(z.array(z.number())),
  attention_patterns: z.record(z.array(z.number())),
  residual_stream: z.array(z.number()),
  tokens: z.array(z.string()),
  metadata: z.record(z.any()),
});

// Feature localization result schema
export const FeatureLocalizationResultSchema = z.object({
  feature_name: z.string(),
  neurons: z.array(z.object({
    layer: z.number(),
    neuron_id: z.number(),
    activation_strength: z.number(),
    confidence: z.number(),
  })),
  validation_metrics: ValidationResultSchema,
});

// Circuit discovery result schema
export const CircuitDiscoveryResultSchema = z.object({
  phenomenon: z.string(),
  circuits: z.array(z.object({
    id: z.string(),
    name: z.string(),
    confidence: z.number(),
    layers: z.array(z.number()),
    components: z.array(z.string()),
    validation_metrics: ValidationResultSchema,
  })),
  execution_time_ms: z.number(),
  model_info: z.object({
    model_id: z.string(),
    architecture: z.string(),
    num_layers: z.number(),
  }),
});

/**
 * Validates an analysis result
 * @param {unknown} result - Raw result object
 * @returns {object} Validated result
 */
export function validateAnalysisResult(result) {
  return AnalysisResultSchema.parse(result);
}

/**
 * Validates activation data
 * @param {unknown} data - Raw activation data
 * @returns {object} Validated activation data
 */
export function validateActivationData(data) {
  return ActivationDataSchema.parse(data);
}
