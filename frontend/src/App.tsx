import { useMemo, useState } from 'react'
import { CategoryFilter } from './components/CategoryFilter'
import { HeatmapPlaceholder } from './components/HeatmapPlaceholder'
import { MetricsHeader } from './components/MetricsHeader'
import { ModelAttackTable } from './components/ModelAttackTable'
import { RunSummaryCards } from './components/RunSummaryCards'
import { mockMetrics } from './data/mockMetrics'
import type { MetricCell } from './types/metrics'

function App() {
  const [category, setCategory] = useState<string>('all')

  const categories = useMemo(
    () => ['all', ...new Set(mockMetrics.cells.map((cell) => cell.category))],
    [],
  )

  const filteredCells = useMemo<MetricCell[]>(() => {
    if (category === 'all') {
      return mockMetrics.cells
    }
    return mockMetrics.cells.filter((cell) => cell.category === category)
  }, [category])

  return (
    <main className="dashboard">
      <MetricsHeader
        title="AIxBio Safeguard Dashboard"
        subtitle="Models x attacks x categories with StrongREJECT-ready scoring placeholders"
      />

      <CategoryFilter
        categories={categories}
        selectedCategory={category}
        onChange={setCategory}
      />

      <RunSummaryCards
        totalCells={filteredCells.length}
        successCount={filteredCells.filter((cell) => cell.succeeded).length}
        avgScore={
          filteredCells.reduce((sum, cell) => sum + cell.score, 0) /
          Math.max(filteredCells.length, 1)
        }
        attackMethods={mockMetrics.attackMethods.length}
      />

      <HeatmapPlaceholder
        cells={filteredCells}
        models={mockMetrics.models}
        methods={mockMetrics.attackMethods}
      />

      <ModelAttackTable cells={filteredCells} />
    </main>
  )
}

export default App
