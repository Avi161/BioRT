interface CategoryFilterProps {
  categories: string[]
  selectedCategory: string
  onChange: (category: string) => void
}

export function CategoryFilter({
  categories,
  selectedCategory,
  onChange,
}: CategoryFilterProps) {
  return (
    <section className="card">
      <label htmlFor="category-filter" className="small-label">
        Category filter
      </label>
      <div style={{ marginTop: '0.4rem' }}>
        <select
          id="category-filter"
          value={selectedCategory}
          onChange={(event) => onChange(event.target.value)}
        >
          {categories.map((category) => (
            <option key={category} value={category}>
              {category}
            </option>
          ))}
        </select>
      </div>
    </section>
  )
}
