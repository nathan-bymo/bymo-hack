import { Navigate, Route, Routes } from 'react-router-dom'
import ComparePage from './pages/ComparePage'
import ImageDetailPage from './pages/ImageDetailPage'
import PerformancePage from './pages/PerformancePage'
import ProductPage from './pages/ProductPage'
import QueuePage from './pages/QueuePage'
import StorefrontPage from './pages/StorefrontPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Navigate to="/product/DEMO-0001" replace />} />
      <Route path="/product/:productId" element={<ProductPage />} />
      <Route path="/image-classification" element={<QueuePage />} />
      <Route path="/image-classification/compare" element={<ComparePage />} />
      <Route path="/image-classification/performance" element={<PerformancePage />} />
      <Route path="/image-classification/:imageId" element={<ImageDetailPage />} />
      <Route path="/storefront-preview" element={<StorefrontPage />} />
      <Route path="*" element={<Navigate to="/product/DEMO-0001" replace />} />
    </Routes>
  )
}
