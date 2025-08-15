/**
 * Circuit types and schemas
 */

import { z } from 'zod';

// Circuit validation status enum
export const ValidationStatusSchema = z.enum(['validated', 'pending', 'failed', 'unknown']);

// Circuit component schema
export const CircuitComponentSchema = z.object({
  type: z.enum(['attention', 'mlp', 'residual', 'embedding']),
  layer: z.number(),
  component_id: z.string(),
  weight: z.number().optional(),
  metadata: z.record(z.any()).optional(),
});

// Circuit schema
export const CircuitSchema = z.object({
  id: z.string(),
  name: z.string(),
  description: z.string(),
  phenomenon: z.string(),
  layers: z.array(z.number()),
  components: z.array(CircuitComponentSchema),
  confidence: z.number().min(0).max(1),
  metadata: z.record(z.any()),
  validation_status: ValidationStatusSchema,
  created_at: z.string(),
  updated_at: z.string(),
});

// Circuit library entry schema
export const CircuitLibraryEntrySchema = z.object({
  circuit: CircuitSchema,
  version: z.string(),
  tags: z.array(z.string()),
  author: z.string().optional(),
  references: z.array(z.string()).optional(),
  performance_metrics: z.record(z.number()).optional(),
});

// Circuit composition schema
export const CircuitCompositionSchema = z.object({
  id: z.string(),
  name: z.string(),
  circuits: z.array(z.string()), // Circuit IDs
  composition_type: z.enum(['sequential', 'parallel', 'hierarchical']),
  dependencies: z.array(z.object({
    from: z.string(),
    to: z.string(),
    type: z.string(),
  })),
  validation_result: z.object({
    valid: z.boolean(),
    conflicts: z.array(z.string()),
    performance_impact: z.number(),
  }).optional(),
});

/**
 * Validates a circuit definition
 * @param {unknown} circuit - Raw circuit object
 * @returns {object} Validated circuit
 */
export function validateCircuit(circuit) {
  return CircuitSchema.parse(circuit);
}

/**
 * Validates a circuit library entry
 * @param {unknown} entry - Raw library entry
 * @returns {object} Validated library entry
 */
export function validateCircuitLibraryEntry(entry) {
  return CircuitLibraryEntrySchema.parse(entry);
}
