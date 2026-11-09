function shortLabel(value) {
  const text = String(value || '')
  return text.length > 18 ? `${text.slice(0, 17)}…` : text
}

function buildPositions(nodes) {
  const query = nodes.filter((node) => node.role === 'query')
  const evidence = nodes.filter((node) => ['predicted', 'candidate'].includes(node.role))
  const context = nodes.filter((node) => !['query', 'predicted', 'candidate'].includes(node.role))
  const positions = {}

  query.forEach((node) => { positions[node.id] = { x: 84, y: 145 } })
  const placeColumn = (items, x) => {
    const gap = Math.min(78, 220 / Math.max(items.length, 1))
    const start = 145 - ((items.length - 1) * gap) / 2
    items.forEach((node, index) => {
      positions[node.id] = { x, y: start + index * gap }
    })
  }
  placeColumn(evidence.length ? evidence : context.slice(0, 1), 335)
  placeColumn(evidence.length ? context : context.slice(1), 585)
  return positions
}

export default function CategoryRelationGraph({ graph, selectedCategory, t }) {
  let nodes = graph?.nodes || []
  let edges = graph?.edges || []

  if (selectedCategory && !nodes.some((node) => node.label === selectedCategory)) {
    const selectedNode = {
      id: `category:${selectedCategory}`,
      label: selectedCategory,
      role: 'result',
      score: null,
    }
    nodes = [...nodes, selectedNode].slice(0, 7)
    const validIds = new Set(nodes.map((node) => node.id))
    edges = edges.filter((edge) => validIds.has(edge.source) && validIds.has(edge.target))
    edges.push({
      source: 'query',
      target: selectedNode.id,
      relation: 'result',
      label: 'Top-K',
    })
  }

  if (!nodes.length) {
    return <div className="graph-empty">{t('explanation.noCategoryEvidence')}</div>
  }

  const positions = buildPositions(nodes)
  return (
    <div className="category-graph-wrap">
      <svg className="category-graph" viewBox="0 0 680 290" role="img" aria-label={t('explanation.graphAria')}>
        <defs>
          <marker id="category-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="currentColor" />
          </marker>
        </defs>
        {edges.map((edge, index) => {
          const source = positions[edge.source]
          const target = positions[edge.target]
          if (!source || !target) return null
          const related = edge.relation === 'mutual_neighbor'
          return (
            <g key={`${edge.source}-${edge.target}-${index}`} className={`graph-edge graph-edge--${edge.relation}`}>
              <line
                x1={source.x + (source.x < target.x ? 62 : -62)}
                y1={source.y}
                x2={target.x + (source.x < target.x ? -62 : 62)}
                y2={target.y}
                markerEnd={related ? undefined : 'url(#category-arrow)'}
              />
              {edge.label && <text x={(source.x + target.x) / 2} y={(source.y + target.y) / 2 - 6}>{edge.label}</text>}
            </g>
          )
        })}
        {nodes.map((node) => {
          const position = positions[node.id]
          if (!position) return null
          const isSelected = selectedCategory && node.label === selectedCategory
          return (
            <g
              key={node.id}
              className={`graph-node graph-node--${node.role}${isSelected ? ' graph-node--selected' : ''}`}
              transform={`translate(${position.x - 62} ${position.y - 24})`}
            >
              <title>{node.label}{node.score != null ? ` · ${Number(node.score).toFixed(3)}` : ''}</title>
              <rect width="124" height="48" rx="5" />
              <text className="graph-node-label" x="62" y="20" textAnchor="middle">{shortLabel(node.label)}</text>
              <text className="graph-node-meta" x="62" y="35" textAnchor="middle">
                {node.role === 'query' ? 'QUERY' : node.score != null ? Number(node.score).toFixed(3) : node.role.toUpperCase()}
              </text>
            </g>
          )
        })}
      </svg>
      <div className="graph-legend">
        <span><i className="legend-dot legend-dot--primary" />{t('explanation.primaryCategory')}</span>
        <span><i className="legend-line legend-line--related" />{t('explanation.relatedCategory')}</span>
        <span><i className="legend-ring" />{t('explanation.selectedProduct')}</span>
      </div>
    </div>
  )
}
