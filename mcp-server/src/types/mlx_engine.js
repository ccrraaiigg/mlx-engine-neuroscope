/**
 * MLX Engine integration types and schemas
 */

import { z } from 'zod';

// MLX Engine API request schema
export const MLXEngineRequestSchema = z.object({
  endpoint: z.string(),
  method: z.enum(['GET', 'POST', 'PUT', 'DELETE']),
  body: z.record(z.any()).optional(),
  headers: z.record(z.string()).optional(),
  timeout: z.number().optional(),
});

// MLX Engine API response schema
export const MLXEngineResponseSchema = z.object({
  success: z.boolean(),
  data: z.any().optional(),
  error: z.string().optional(),
  status_code: z.number(),
  request_id: z.string(),
});

// Model load result schema
export const ModelLoadResultSchema = z.object({
  model_id: z.string(),
  model_path: z.string(),
  architecture: z.string(),
  num_layers: z.number(),
  vocab_size: z.number(),
  loaded_at: z.string(),
});

// Activation hook specification schema
export const ActivationHookSpecSchema = z.object({
  layer_name: z.string(),
  component: z.string(),
  hook_id: z.string(),
  capture_input: z.boolean(),
  capture_output: z.boolean(),
});

// Captured activation schema
export const CapturedActivationSchema = z.object({
  hook_id: z.string(),
  layer_name: z.string(),
  component: z.string(),
  shape: z.array(z.number()),
  data: z.array(z.number()),
  timestamp: z.string(),
});

// Generation with activations schema
export const GenerationWithActivationsSchema = z.object({
  choices: z.array(z.object({
    message: z.object({
      content: z.string(),
    }),
    finish_reason: z.string(),
  })),
  activations: z.record(z.array(CapturedActivationSchema)),
  usage: z.object({
    prompt_tokens: z.number(),
    completion_tokens: z.number(),
    total_tokens: z.number(),
  }),
});

// Streaming result schema
export const StreamingResultSchema = z.object({
  type: z.enum(['token', 'activation', 'done']),
  data: z.any(),
  timestamp: z.string(),
});

/**
 * Validates an MLX Engine request
 * @param {unknown} request - Raw request object
 * @returns {object} Validated request
 */
export function validateMLXEngineRequest(request) {
  return MLXEngineRequestSchema.parse(request);
}

/**
 * Validates an MLX Engine response
 * @param {unknown} response - Raw response object
 * @returns {object} Validated response
 */
export function validateMLXEngineResponse(response) {
  return MLXEngineResponseSchema.parse(response);
}
