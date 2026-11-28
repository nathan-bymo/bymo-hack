import { Navigate, Route, Routes } from 'react-router-dom'
import ClassifyPage from './pages/ClassifyPage'
import HowPage from './pages/HowPage'
import ResultsPage from './pages/ResultsPage'

/** Tre pagine, nell'ordine del video: la demo, i numeri, come funziona. */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ClassifyPage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/how" element={<HowPage />} />
      <Route path="/models" element={<Navigate to="/results" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
