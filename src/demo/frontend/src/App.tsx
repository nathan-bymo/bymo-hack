import { Navigate, Route, Routes } from 'react-router-dom'
import ClassifyPage from './pages/ClassifyPage'
import OverviewPage from './pages/OverviewPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ClassifyPage />} />
      <Route path="/models" element={<OverviewPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
