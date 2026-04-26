interface RunSummaryCardsProps {
  totalCells: number
  successCount: number
  avgScore: number
  attackMethods: number
}

export function RunSummaryCards({
  totalCells,
  successCount,
  avgScore,
  attackMethods,
}: RunSummaryCardsProps) {
  return (
    <section className="summary-grid">
      <article className="card">
        <div className="small-label">Visible cells</div>
        <div className="metric-value">{totalCells}</div>
      </article>
      <article className="card">
        <div className="small-label">Successes</div>
        <div className="metric-value">{successCount}</div>
      </article>
      <article className="card">
        <div className="small-label">Avg score</div>
        <div className="metric-value">{avgScore.toFixed(2)}</div>
      </article>
      <article className="card">
        <div className="small-label">Attack methods</div>
        <div className="metric-value">{attackMethods}</div>
      </article>
    </section>
  )
}
