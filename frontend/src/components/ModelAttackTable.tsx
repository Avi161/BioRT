import type { MetricCell } from '../types/metrics'

interface ModelAttackTableProps {
  cells: MetricCell[]
}

export function ModelAttackTable({ cells }: ModelAttackTableProps) {
  const sorted = [...cells].sort((a, b) => {
    if (a.model !== b.model) return a.model.localeCompare(b.model)
    if (a.method !== b.method) return a.method.localeCompare(b.method)
    if (a.category !== b.category) return a.category.localeCompare(b.category)
    return a.promptId.localeCompare(b.promptId)
  })

  return (
    <section className="card">
      <h2 style={{ marginTop: 0 }}>Cell Details</h2>
      <div className="table-wrap">
        <table className="metrics-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Method</th>
              <th>Category</th>
              <th>Prompt ID</th>
              <th>Score</th>
              <th>Succeeded</th>
              <th>Latency (s)</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((cell) => (
              <tr key={`${cell.model}-${cell.method}-${cell.promptId}`}>
                <td>{cell.model}</td>
                <td>{cell.method}</td>
                <td>{cell.category}</td>
                <td>{cell.promptId}</td>
                <td>{cell.score.toFixed(2)}</td>
                <td>{cell.succeeded ? 'yes' : 'no'}</td>
                <td>{cell.latencySeconds.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
