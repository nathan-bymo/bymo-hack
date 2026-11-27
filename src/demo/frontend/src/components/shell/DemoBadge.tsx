/** Badge permanente: non chiudibile, sempre visibile (D.2). */
export function DemoBadge() {
  return (
    <span
      data-testid="demo-badge"
      className="inline-flex items-center rounded-full border border-white/60 bg-white/10 px-3 py-1 text-xs font-semibold text-titlebar-fg"
    >
      Demo — Team Bymo · Hack Apertus
    </span>
  )
}
