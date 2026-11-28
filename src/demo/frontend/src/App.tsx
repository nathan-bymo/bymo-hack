import { Navigate, Route, Routes } from 'react-router-dom'
import CatalogPage from './pages/CatalogPage'
import CatalogProductPage from './pages/CatalogProductPage'
import ClassifyPage from './pages/ClassifyPage'
import HowPage from './pages/HowPage'
import ResultsPage from './pages/ResultsPage'
import ShopPage from './pages/ShopPage'
import ShopProductPage from './pages/ShopProductPage'

/** Le due facce della demo (vetrina e catalogo) più le pagine tecniche. */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ClassifyPage />} />
      <Route path="/shop" element={<ShopPage />} />
      <Route path="/shop/:productId" element={<ShopProductPage />} />
      <Route path="/catalog" element={<CatalogPage />} />
      <Route path="/catalog/:productId" element={<CatalogProductPage />} />
      <Route path="/results" element={<ResultsPage />} />
      <Route path="/how" element={<HowPage />} />
      <Route path="/models" element={<Navigate to="/results" replace />} />
      <Route path="/image-classification" element={<Navigate to="/" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
