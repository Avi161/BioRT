import type { AttackMethod, MetricCell } from '../types/metrics'

interface HeatmapPlaceholderProps {
  models: string[]
  methods: AttackMethod[]
  cells: MetricCell[]
}

function scoreToColor(score: number): string {
  const clamped = Math.max(0, Math.min(10, score))
  const red = Math.round((clamped / 10) * 220)
  const green = Math.round(((10 - clamped) / 10) * 220)
  return `rgb(${red}, ${green}, 120)`
}

export function HeatmapPlaceholder({
  models,
  methods,
  cells,
}: HeatmapPlaceholderProps) {
  const avgByModelAndMethod = new Map<string, number>()

  for (const model of models) {
    for (const method of methods) {
      const bucket = cells.filter((cell) => cell.model === model && cell.method === method)
      const avg =
        bucket.reduce((sum, cell) => sum + cell.score, 0) / Math.max(bucket.length, 1)
      avgByModelAndMethod.set(`${model}::${method}`, avg)
    }
  }

  return (
    <section className="card">
      <h2 style={{ marginTop: 0 }}>Heatmap (placeholder)</h2>
      <p className="small-label" style={{ marginBottom: '0.75rem' }}>
        Average score by model x attack method for the currently selected category filter.
      </p>
      <div className="heatmap-grid" style={{ ['--method-count' as string]: methods.length }}>
        <div className="heatmap-row">
          <div className="heatmap-label small-label">Model / Method</div>
          {methods.map((method) => (
            <div key={method} className="heatmap-label small-label">
              {method}
            </div>
          ))}
        </div>
        {models.map((model) => (
          <div key={model} className="heatmap-row">
            <div className="heatmap-label">{model}</div>
            {methods.map((method) => {
              const key = `${model}::${method}`
              const avg = avgByModelAndMethod.get(key) ?? 0
              return (
                <div
                  key={key}
                  className="heatmap-cell"
                  style={{ backgroundColor: scoreToColor(avg), color: '#111827' }}
                >
                  {avg.toFixed(2)}
                </div>
              )
            })}
          </div>
        ))}
      </div>
    </section>
  )
}
