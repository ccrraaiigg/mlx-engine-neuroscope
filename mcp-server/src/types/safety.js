/**
 * Safety and alignment types and schemas
 */

import { z } from 'zod';

// Risk level enum
export const RiskLevelSchema = z.enum(['low', 'medium', 'high', 'critical']);

// Safety assessment schema
export const SafetyAssessmentSchema = z.object({
  operation: z.string(),
  risk_level: RiskLevelSchema,
  risk_factors: z.array(z.object({
    factor: z.string(),
    severity: z.number().min(0).max(1),
    description: z.string(),
  })),
  mitigation_strategies: z.array(z.string()),
  approval_required: z.boolean(),
  timestamp: z.string(),
});

// Safety intervention schema
export const SafetyInterventionSchema = z.object({
  id: z.string(),
  type: z.enum(['circuit_disable', 'activation_clamp', 'weight_freeze', 'output_filter']),
  target: z.string(),
  parameters: z.record(z.any()),
  effectiveness: z.number().min(0).max(1).optional(),
  side_effects: z.array(z.string()).optional(),
});

// Harmful circuit detection result schema
export const HarmfulCircuitDetectionSchema = z.object({
  circuits: z.array(z.object({
    id: z.string(),
    name: z.string(),
    harm_category: z.enum(['bias', 'toxicity', 'misinformation', 'manipulation', 'other']),
    severity: z.number().min(0).max(1),
    confidence: z.number().min(0).max(1),
    evidence: z.array(z.string()),
  })),
  recommended_interventions: z.array(SafetyInterventionSchema),
  assessment: SafetyAssessmentSchema,
});

// Post-modification validation schema
export const PostModificationValidationSchema = z.object({
  modification_id: z.string(),
  performance_metrics: z.object({
    accuracy_retention: z.number().min(0).max(1),
    capability_preservation: z.record(z.number()),
    safety_improvement: z.number().min(0).max(1),
  }),
  safety_metrics: z.object({
    harm_reduction: z.number().min(0).max(1),
    false_positive_rate: z.number().min(0).max(1),
    intervention_effectiveness: z.number().min(0).max(1),
  }),
  validation_status: z.enum(['passed', 'failed', 'warning']),
  recommendations: z.array(z.string()),
});

/**
 * Validates a safety assessment
 * @param {unknown} assessment - Raw assessment object
 * @returns {object} Validated assessment
 */
export function validateSafetyAssessment(assessment) {
  return SafetyAssessmentSchema.parse(assessment);
}

/**
 * Validates a safety intervention
 * @param {unknown} intervention - Raw intervention object
 * @returns {object} Validated intervention
 */
export function validateSafetyIntervention(intervention) {
  return SafetyInterventionSchema.parse(intervention);
}
