export interface CrescendoConversationTurn {
  crescendo_round?: number
  turn_index: number
  role: string | null
  text: string | null
  data_type: string | null
  original_value: string | null
  converted_value: string | null
}

export interface CrescendoTraceMetadata {
  prompt_file: string
  prompt_id: string | null
  category: string
  method: 'crescendo'
  /** Registry provider id, e.g. `deepseek` — groups runs into one file per defender. */
  defender_provider?: string
  /** Registry provider id for the adversarial (attacker) model, e.g. `moonshot`. */
  attacker_provider?: string
  conversation_id: string
  status: string
  created_at_utc: string
  /** Kimi (attacker) → defender prompt count, excluding PyRIT internal user lines. */
  total_turns: number
  attacker_turns: number
  defender_turns: number
  /** Defender replies classified as refusals (Crescendo backtrack trigger), not HTTP retries. */
  defender_refusals: number
  /** Raw message pieces from PyRIT memory (includes scaffolding). */
  raw_transcript_pieces?: number
  /** Matrix / transport retries for the cell, not Crescendo semantic backtracks. */
  http_cell_attempts?: number
  http_cell_retries?: number
  /** @deprecated Old field; prefer http_cell_* */
  attempts_made?: number
  /** @deprecated Old field; was HTTP retries — prefer defender_refusals for Crescendo. */
  total_retries?: number
  elapsed_seconds: number | null
}

export interface CrescendoTraceRecord {
  metadata: CrescendoTraceMetadata
  defender_model: string
  attacker_model: string
  objective_prompt: string
  attacker_conversation: CrescendoConversationTurn[]
  defender_conversation: CrescendoConversationTurn[]
}
