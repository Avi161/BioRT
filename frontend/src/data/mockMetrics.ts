import type { AttackMethod, MetricCell, MetricsDataset } from '../types/metrics'

const models = ['DeepSeek V4 Flash', 'Kimi K2.5', 'Gemini 3 Pro']
const attackMethods: AttackMethod[] = ['direct', 'base64', 'pair', 'crescendo']
const categories = [
  'synthesis_evasion',
  'lab_automation_uplift',
  'dual_use_protocol',
  'acquisition_kyc',
  'virology',
]

const cells: MetricCell[] = []

for (const model of models) {
  for (const method of attackMethods) {
    for (const category of categories) {
      for (let i = 1; i <= 3; i += 1) {
        const scoreBase = (model.length + method.length + category.length + i) % 10
        cells.push({
          model,
          method,
          category,
          promptId: `${category}-${String(i).padStart(2, '0')}`,
          score: scoreBase,
          succeeded: scoreBase >= 6,
          latencySeconds: Number((1 + scoreBase / 3).toFixed(2)),
        })
      }
    }
  }
}

export const mockMetrics: MetricsDataset = {
  models,
  attackMethods,
  categories,
  cells,
}
