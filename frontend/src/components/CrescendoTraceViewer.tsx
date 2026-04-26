import { useMemo, useState } from 'react'
import type { ChangeEvent } from 'react'

import { parseCrescendoTraceJsonl } from '../lib/parseCrescendoTrace'
import type {
  CrescendoConversationTurn,
  CrescendoTraceRecord,
} from '../types/crescendoTrace'

type TimelineTurn = {
  side: 'attacker' | 'defender'
  model: string
  turn: CrescendoConversationTurn
}

function getTurnText(turn: CrescendoConversationTurn): string {
  return turn.text ?? turn.converted_value ?? turn.original_value ?? ''
}

function toPlainText(markdownText: string): string {
  return markdownText
    .replace(/```[\s\S]*?```/g, (block) => block.replace(/```/g, ''))
    .replace(/`([^`]+)`/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/^#{1,6}\s*/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/^\s*\d+\.\s+/gm, '')
    .replace(/\*\*([^*]+)\*\*/g, '$1')
    .replace(/\*([^*]+)\*/g, '$1')
    .replace(/_([^_]+)_/g, '$1')
}

function toTimeline(record: CrescendoTraceRecord): TimelineTurn[] {
  const attackerTurns: TimelineTurn[] = record.attacker_conversation.map((turn) => ({
    side: 'attacker',
    model: record.attacker_model,
    turn,
  }))
  const defenderTurns: TimelineTurn[] = record.defender_conversation.map((turn) => ({
    side: 'defender',
    model: record.defender_model,
    turn,
  }))

  return [...attackerTurns, ...defenderTurns].sort(
    (a, b) => a.turn.turn_index - b.turn.turn_index,
  )
}

export function CrescendoTraceViewer() {
  const [records, setRecords] = useState<CrescendoTraceRecord[]>([])
  const [error, setError] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')

  const totalTurns = useMemo(
    () => records.reduce((sum, record) => sum + record.metadata.total_turns, 0),
    [records],
  )

  async function handleFileUpload(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) {
      return
    }

    setFileName(file.name)
    try {
      const contents = await file.text()
      const parsed = parseCrescendoTraceJsonl(contents)
      setRecords(parsed)
      setError(null)
    } catch (uploadError) {
      setRecords([])
      setError(uploadError instanceof Error ? uploadError.message : 'Could not parse trace file.')
    }
  }

  return (
    <section className="card">
      <h2 style={{ marginTop: 0 }}>Crescendo Transcript Viewer</h2>
      <p className="small-label" style={{ marginTop: 0 }}>
        Upload a Crescendo trace JSONL file to inspect attacker/defender turns with stable formatting.
      </p>

      <label className="upload-label">
        Upload trace file
        <input type="file" accept=".jsonl,.json,application/json" onChange={handleFileUpload} />
      </label>

      {fileName ? (
        <p className="small-label">
          Loaded: <strong>{fileName}</strong> | Runs: <strong>{records.length}</strong> | Kimi
          prompts (sum): <strong>{totalTurns}</strong>
        </p>
      ) : null}

      {error ? <p className="trace-error">{error}</p> : null}

      <div className="trace-records">
        {records.map((record) => (
          <article className="trace-record" key={record.metadata.conversation_id}>
            <header className="trace-record-header">
              <h3 style={{ margin: 0 }}>
                {record.attacker_model} (attacker) vs {record.defender_model} (defender)
              </h3>
              <p className="small-label" style={{ margin: 0 }}>
                Category: {record.metadata.category} | Prompt: {record.metadata.prompt_id ?? 'no-id'} |
                Status: {record.metadata.status}
              </p>
              <p className="small-label" style={{ margin: 0 }}>
                Kimi prompts: {record.metadata.total_turns} (defender replies{' '}
                {record.metadata.defender_turns}) | Refusals (backtrack trigger):{' '}
                {record.metadata.defender_refusals}
                {record.metadata.raw_transcript_pieces != null
                  ? ` | Raw memory pieces: ${record.metadata.raw_transcript_pieces}`
                  : null}
                {record.metadata.http_cell_retries != null && record.metadata.http_cell_retries > 0
                  ? ` | HTTP cell retries: ${record.metadata.http_cell_retries}`
                  : null}
              </p>
            </header>

            <section className="trace-prompt-objective">
              <h4>Objective Prompt</h4>
              <p>{toPlainText(record.objective_prompt)}</p>
            </section>

            <section className="trace-turn-list">
              {toTimeline(record).map(({ side, model, turn }) => {
                const isAttackerTurn = side === 'attacker'
                return (
                  <div
                    className={`trace-turn ${isAttackerTurn ? 'trace-turn-attacker' : 'trace-turn-defender'}`}
                    key={`${record.metadata.conversation_id}-${turn.turn_index}-${side}`}
                  >
                    <h4>
                      {turn.crescendo_round != null
                        ? `Round ${turn.crescendo_round}`
                        : `Step ${turn.turn_index}`}
                      : {isAttackerTurn ? 'Attacker' : 'Defender'} ({model})
                    </h4>
                    <p>{toPlainText(getTurnText(turn))}</p>
                  </div>
                )
              })}
            </section>
          </article>
        ))}
      </div>
    </section>
  )
}
