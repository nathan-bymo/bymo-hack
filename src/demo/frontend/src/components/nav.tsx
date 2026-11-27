import { NavLink, useSearchParams } from 'react-router-dom'

/** Schede comuni delle pagine "Image classification (AI)": la sessione resta nell'URL. */
export function DemoTabs() {
  const [params] = useSearchParams()
  const session = params.get('session')
  const q = session ? `?session=${session}` : ''
  const tabs = [
    { to: `/image-classification${q}`, path: '/image-classification', label: 'Queue' },
    { to: `/image-classification/try${q}`, path: '/image-classification/try', label: 'Try a model' },
    { to: `/image-classification/compare${q}`, path: '/image-classification/compare', label: 'Compare (batch)' },
    { to: `/image-classification/models${q}`, path: '/image-classification/models', label: 'Models' },
    { to: '/image-classification/performance?set=coverage', path: '/image-classification/performance', label: 'Per-class metrics' },
  ]
  return (
    <nav aria-label="Image classification" className="flex gap-1 border-b border-line bg-panel px-3 pt-1 text-sm" data-testid="demo-tabs">
      {tabs.map((t) => (
        <NavLink key={t.path} to={t.to} end
          className={({ isActive }) => `-mb-px rounded-t border px-3 py-1 no-underline ${isActive
            ? 'border-line border-b-panel bg-panel font-semibold text-ink'
            : 'border-transparent text-link hover:bg-app'}`}>
          {t.label}
        </NavLink>
      ))}
    </nav>
  )
}
