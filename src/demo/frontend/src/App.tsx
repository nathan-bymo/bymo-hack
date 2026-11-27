import { Navigate, Route, Routes } from 'react-router-dom'
import ComparePage from './pages/ComparePage'
import ImageDetailPage from './pages/ImageDetailPage'
import ClassifyPage from './pages/ClassifyPage'
import OverviewPage from './pages/OverviewPage'
import PerformancePage from './pages/PerformancePage'
import ProductPage from './pages/ProductPage'
import QueuePage from './pages/QueuePage'
import StorefrontPage from './pages/StorefrontPage'

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<ClassifyPage />} />
      <Route path="/models" element={<OverviewPage />} />
      <Route path="/product/:productId" element={<ProductPage />} />
      <Route path="/image-classification" element={<QueuePage />} />
      <Route path="/image-classification/try" element={<Navigate to="/" replace />} />
      <Route path="/image-classification/models" element={<Navigate to="/models" replace />} />
      <Route path="/image-classification/compare" element={<ComparePage />} />
      <Route path="/image-classification/performance" element={<PerformancePage />} />
      <Route path="/image-classification/:imageId" element={<ImageDetailPage />} />
      <Route path="/storefront-preview" element={<StorefrontPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
