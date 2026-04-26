export type AttackMethod = 'direct' | 'base64' | 'pair' | 'crescendo'

export interface MetricCell {
  model: string
  method: AttackMethod
  category: string
  promptId: string
  score: number
  succeeded: boolean
  latencySeconds: number
}

export interface MetricsDataset {
  models: string[]
  attackMethods: AttackMethod[]
  categories: string[]
  cells: MetricCell[]
}

export interface BackendResultRecord {
  model?: { display_name?: string; provider?: string }
  attack?: { method?: string }
  outcome?: string
  elapsed_seconds?: number
  conversation_id?: string | null
}
