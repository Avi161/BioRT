interface MetricsHeaderProps {
  title: string
  subtitle: string
}

export function MetricsHeader({ title, subtitle }: MetricsHeaderProps) {
  return (
    <section className="card">
      <h1 style={{ margin: 0 }}>{title}</h1>
      <p className="small-label" style={{ marginTop: '0.5rem' }}>
        {subtitle}
      </p>
    </section>
  )
}
