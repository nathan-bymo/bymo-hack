import { Navigate, Route, Routes } from 'react-router-dom'
import CatalogPage from './pages/CatalogPage'
import ChallengePage from './pages/ChallengePage'
import CatalogProductPage from './pages/CatalogProductPage'
import ClassifyPage from './pages/ClassifyPage'
import ShopPage from './pages/ShopPage'
import ShopProductPage from './pages/ShopProductPage'

/** Le due facce della demo (vetrina e catalogo) e la pagina di classificazione. I vecchi indirizzi portano a `/`. */
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ClassifyPage />} />
      <Route path="/challenge" element={<ChallengePage />} />
      <Route path="/shop" element={<ShopPage />} />
      <Route path="/shop/:productId" element={<ShopProductPage />} />
      <Route path="/catalog" element={<CatalogPage />} />
      <Route path="/catalog/:productId" element={<CatalogProductPage />} />
      <Route path="/image-classification" element={<Navigate to="/" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
