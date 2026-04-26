import type { CrescendoTraceMetadata, CrescendoTraceRecord } from '../types/crescendoTrace'

function isConversationTurn(value: unknown): boolean {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const turn = value as Record<string, unknown>
  return (
    typeof turn.turn_index === 'number' &&
    ('role' in turn) &&
    ('text' in turn) &&
    ('data_type' in turn)
  )
}

function isTraceRecordLoose(value: unknown): value is CrescendoTraceRecord {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const record = value as Record<string, unknown>
  const metadata =
    typeof record.metadata === 'object' && record.metadata !== null
      ? (record.metadata as Record<string, unknown>)
      : null

  return (
    metadata !== null &&
    typeof metadata.prompt_file === 'string' &&
    typeof metadata.category === 'string' &&
    (typeof metadata.prompt_id === 'string' || metadata.prompt_id === null) &&
    typeof metadata.method === 'string' &&
    typeof metadata.conversation_id === 'string' &&
    typeof metadata.status === 'string' &&
    typeof metadata.total_turns === 'number' &&
    typeof record.defender_model === 'string' &&
    typeof record.attacker_model === 'string' &&
    typeof record.objective_prompt === 'string' &&
    Array.isArray(record.attacker_conversation) &&
    Array.isArray(record.defender_conversation) &&
    record.attacker_conversation.every((turn) => isConversationTurn(turn)) &&
    record.defender_conversation.every((turn) => isConversationTurn(turn))
  )
}

function normalizeTraceMetadata(
  raw: Record<string, unknown> | CrescendoTraceMetadata,
): CrescendoTraceMetadata {
  const m = { ...raw } as Record<string, unknown>
  if (typeof m.defender_refusals !== 'number') {
    m.defender_refusals = 0
  }
  if (typeof m.http_cell_attempts !== 'number' && typeof m.attempts_made === 'number') {
    m.http_cell_attempts = m.attempts_made
  }
  if (typeof m.http_cell_retries !== 'number' && typeof m.total_retries === 'number') {
    m.http_cell_retries = m.total_retries
  }
  return m as unknown as CrescendoTraceMetadata
}

function normalizeTraceRecord(record: CrescendoTraceRecord): CrescendoTraceRecord {
  return {
    ...record,
    metadata: normalizeTraceMetadata(record.metadata as unknown as Record<string, unknown>),
  }
}

function isTraceRecord(value: unknown): value is CrescendoTraceRecord {
  return isTraceRecordLoose(value)
}

function isLegacyTraceRecord(value: unknown): boolean {
  if (typeof value !== 'object' || value === null) {
    return false
  }
  const record = value as Record<string, unknown>
  return (
    typeof record.prompt_file === 'string' &&
    typeof record.category === 'string' &&
    typeof record.prompt === 'string' &&
    typeof record.defender_model === 'string' &&
    typeof record.attacker_model === 'string' &&
    typeof record.status === 'string' &&
    typeof record.conversation_id === 'string' &&
    Array.isArray(record.transcript)
  )
}

function normalizeLegacyRecord(value: unknown): CrescendoTraceRecord {
  const legacy = value as Record<string, unknown>
  const transcript = (legacy.transcript as Array<Record<string, unknown>>) ?? []

  const attackerConversation = transcript
    .map((turn, idx) => ({
      turn_index: idx + 1,
      role: (turn.role as string | null) ?? null,
      text: ((turn.converted_value as string | null) ?? (turn.original_value as string | null)) ?? null,
      data_type: (turn.data_type as string | null) ?? null,
      original_value: (turn.original_value as string | null) ?? null,
      converted_value: (turn.converted_value as string | null) ?? null,
    }))
    .filter((turn) => turn.role !== 'assistant')

  const defenderConversation = transcript
    .map((turn, idx) => ({
      turn_index: idx + 1,
      role: (turn.role as string | null) ?? null,
      text: ((turn.converted_value as string | null) ?? (turn.original_value as string | null)) ?? null,
      data_type: (turn.data_type as string | null) ?? null,
      original_value: (turn.original_value as string | null) ?? null,
      converted_value: (turn.converted_value as string | null) ?? null,
    }))
    .filter((turn) => turn.role === 'assistant')

  return {
    metadata: normalizeTraceMetadata({
      prompt_file: legacy.prompt_file,
      prompt_id: (legacy.prompt_id as string | null) ?? null,
      category: legacy.category,
      method: 'crescendo',
      conversation_id: legacy.conversation_id,
      status: legacy.status,
      created_at_utc: '',
      total_turns: transcript.length,
      attacker_turns: attackerConversation.length,
      defender_turns: defenderConversation.length,
      defender_refusals: 0,
      http_cell_attempts: 1,
      http_cell_retries: 0,
      elapsed_seconds: null,
    } as unknown as Record<string, unknown>),
    defender_model: legacy.defender_model as string,
    attacker_model: legacy.attacker_model as string,
    objective_prompt: legacy.prompt as string,
    attacker_conversation: attackerConversation,
    defender_conversation: defenderConversation,
  }
}

export function parseCrescendoTraceJsonl(contents: string): CrescendoTraceRecord[] {
  const trimmed = contents.trim()
  let parsedJson: unknown
  let parsedAsJson = false
  try {
    parsedJson = JSON.parse(trimmed) as unknown
    parsedAsJson = true
  } catch {
    parsedAsJson = false
  }

  if (parsedAsJson) {
    if (Array.isArray(parsedJson)) {
      const invalidIndex = parsedJson.findIndex(
        (record) => !isTraceRecord(record) && !isLegacyTraceRecord(record),
      )
      if (invalidIndex >= 0) {
        throw new Error(`Array item ${invalidIndex + 1} does not match Crescendo trace format.`)
      }
      return parsedJson.map((record) =>
        isTraceRecord(record) ? normalizeTraceRecord(record) : normalizeLegacyRecord(record),
      )
    }
    if (isTraceRecord(parsedJson)) {
      return [normalizeTraceRecord(parsedJson)]
    }
    if (isLegacyTraceRecord(parsedJson)) {
      return [normalizeLegacyRecord(parsedJson)]
    }
    throw new Error('JSON payload does not match Crescendo trace format.')
  }

  const lines = contents
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

  const parsed = lines.map((line, idx) => {
    try {
      return JSON.parse(line) as unknown
    } catch (error) {
      throw new Error(`Line ${idx + 1} is not valid JSON.`)
    }
  })

  const invalidIndex = parsed.findIndex(
    (record) => !isTraceRecord(record) && !isLegacyTraceRecord(record),
  )
  if (invalidIndex >= 0) {
    throw new Error(`Line ${invalidIndex + 1} does not match Crescendo trace format.`)
  }

  return parsed.map((record) =>
    isTraceRecord(record) ? normalizeTraceRecord(record) : normalizeLegacyRecord(record),
  )
}
