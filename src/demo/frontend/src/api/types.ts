export interface ApiErrorBody { code: string; message: string; details?: Record<string, unknown>; request_id?: string }

export interface TaxonomyClass { index: number; code: string; value: string; erp_label: string; erp_label_source: string }
export interface TaxonomyDimension { key: string; erp_label: string; classes: TaxonomyClass[] }
export interface Taxonomy {
  dimensions: TaxonomyDimension[]
  confidence_bins: { low_below: number; high_from: number; source: string }
}

export interface TopK { label: string; p: number }
export interface DimensionResult {
  label: string | null
  erp_label: string
  erp_label_source: string
  confidence: number | null
  top_k: TopK[] | null
  confidence_method: string | null
  applicable: boolean
}
export interface RecordedInfo { run_id: string; recorded_at: string | null; gpu_name: string | null; images_per_second: number | null }
export interface PredictionResult {
  image_id: string
  prediction_id: string
  model_id: string
  model_revision: string
  cached: boolean
  mode: 'live' | 'recorded'
  valid: boolean
  dimensions: Record<string, DimensionResult>
  raw_output: string | null
  timing: { preprocess_ms: number | null; model_ms: number | null; total_ms: number | null }
  recorded: RecordedInfo | null
  use_context: boolean
  created_at: string
}

export interface ImageItem {
  image_id: string
  session_id: string
  original_filename: string
  sha256: string
  format: string
  mime: string
  width: number
  height: number
  bytes: number
  source_kind: string
  context: Record<string, string>
  context_hash: string
  duplicate: boolean
  file_url: string
  thumbnail_url: string
  review_status?: 'reviewed' | 'corrected' | null
  review_changed?: string[]
  review_values?: Record<string, string> | null
  predictions?: PredictionResult[]
  review?: Review | null
}
export interface Review {
  review_id: string
  image_id: string
  prediction_id: string | null
  model_id: string | null
  status: 'reviewed' | 'corrected'
  values: Record<string, string>
  changed_dimensions: string[]
  created_at: string
}
export interface UploadResponse { images: ImageItem[]; rejected: { original_filename: string; error: ApiErrorBody }[] }

export interface Session { session_id: string; name: string; created_at: string; image_count: number }

export interface EvalSummary { set: string; macro_f1: number | null; validity: string; provenance: string; source: string }
export interface ModelSummary {
  model_id: string
  display_name: string
  family: string
  runtime: string
  status: 'unavailable' | 'not_loaded' | 'loading' | 'ready' | 'unloading' | 'error'
  status_reason: string | null
  modes: ('live' | 'recorded')[]
  recorded_runs: { run_id: string; recorded_at: string; gpu_name: string; images: number }[]
  confidence: { kind: string; method: string } | null
  training: { data: string; phase: number | null; seen_sets?: string[] } | null
  short_name: string
  description: string | null
  training_label: string | null
  featured: boolean
  weights: { repo?: string | null; revision?: string | null; subfolder?: string | null; size_gb?: number | null; local_path?: string | null; location_label?: string | null } | null
  memory: { gb: number | null; kind: 'measured' | 'estimate' | 'not_reported'; what: string; source: string } | null
  speeds: SpeedInfo[]
  scores: ScoreInfo[]
  seen_sets: string[]
  contamination_note: string | null
  vram_estimate_gb: number | null
  headline_eval: EvalSummary | null
  kaggle_dev: { score: number | null; submitted_at?: string | null; note?: string }
  parity: { verified: boolean; scope: string }
}
export interface SpeedInfo {
  images_per_second: number; what: string; conditions: string | null; gpu_label: string | null
  usd_per_hour: number | null; cost_per_100k_usd: number | null; cost_note: string | null; source: string; preferred?: boolean
}
export interface ScoreInfo {
  set: string; macro_f1: number | null; weighted_f1?: number | null; validity: string; source: string
  source_label?: string | null; note?: string | null; checkpoint?: string | null
}
export interface SamplePhoto {
  sample_id: string
  title: string
  filename: string
  file_url: string
  sha256: string
  recorded_models: string[]   // modelli con una risposta già registrata per questa foto: rispondono subito
  reference_labels: Record<string, { label: string; human: boolean }> | null   // etichette di train.csv (foto di train), non di un modello
}
export interface KaggleReference { name: string; score: number; source: string }
export interface ModelsResponse {
  models: ModelSummary[]
  kaggle_references: KaggleReference[]
  reference_line: { value: number; label: string; source: string }
  throughput_target_images_per_second: number
}

export interface BatchInfo {
  images: number
  seconds: number
  images_per_second: number | null
  ms_per_image: number | null
  gpu_name: string
  batch_size: number
  peak_memory_gb: number | null
  recorded: RecordedInfo | null
}
export interface JobSnapshot {
  job_id: string
  model_id: string
  status: 'running' | 'done' | 'error' | 'canceled'
  done: number
  total: number
  results: PredictionResult[]
  errors: { image_id: string | null; error: ApiErrorBody }[]
  batch: BatchInfo | null
  error: ApiErrorBody | null
}
export interface PredictResponse { job_id: string; model_id: string; total: number; cached: number; queued: number; events_url: string }

export interface Settings {
  gpu_label: string
  usd_per_hour: number
  presentation_mode: boolean
  show_ground_truth: boolean
  default_use_context: boolean
  zero_shot_parallel_images: number
  storefront_rule?: StorefrontRule
}
export interface Health { status: string; version: string; git_commit: string | null; gpu_local: boolean; guard: { loaded: boolean; entries: number } }

/* ---- metriche (allegato C, C.7) ---- */
export interface PerClassEntry { precision?: number | null; recall?: number | null; f1?: number | null; support?: number | null }
export interface ConfusionData { labels: string[]; columns: string[]; matrix: number[][]; set: string; source: string }
export interface MetricResult {
  eval_id: string
  validity: 'held_out' | 'phase1_internal' | 'seen_in_training' | 'na'
  provenance: 'recomputed' | 'offline_script' | 'reported_only' | 'kaggle_user' | 'na'
  badges: string[]
  badge_tooltip: string | null
  source: string
  checkpoint: string | null
  trained_on: string | null
  macro_f1: number | null
  weighted_f1: number | null
  per_dimension: Record<string, { macro_f1: number | null; weighted_f1: number | null; support: number | null }>
  per_class: Record<string, Record<string, PerClassEntry>> | null
  per_class_source: string | null
  confusion: Record<string, ConfusionData> | null
  confidence_calibration: unknown
  notes: string[]
}
export interface MetricBlock {
  model_id: string
  set: string
  display_name: string
  results: MetricResult[]
  excluded_invalid: number
  contamination_note: string | null
  training: { data: string; phase: number | null } | null
  kaggle_dev: { score: number | null; badge: string; submitted_at?: string | null; source?: string | null; note?: string | null }
}
export interface Throughput {
  target_images_per_second: number
  measured_in_demo: { job_id: string; model_id: string; images_per_second: number; ms_per_image: number; gpu_name: string; batch_size: number; mode: 'live' | 'recorded'; cost_per_1k_images_usd: number | null; cost_note: string | null }[]
  reported: { model_id: string; value: number; what: string; conditions: string | null; source: string; gpu_label: string | null; usd_per_hour: number | null; cost_per_1k_images_usd: number | null }[]
  gpu_prices: Record<string, number>
}

export interface StorefrontRule {
  label: string
  idea_source: string
  hide_if: Record<string, string[]>
  rank: { name: string; when: Record<string, string[]> }[]
}

/* ---- catalogo e vetrina (lato admin e lato cliente) ---- */
export interface CompactImage {
  image_id: string
  file_url: string
  thumbnail_url: string
  original_filename: string
  width: number
  height: number
  values: Record<string, string | null>
  confidence: number | null
  is_cover: boolean
  position: number
}
export interface CatalogImageState {
  product_id: string
  position: number
  published: boolean
  is_cover: boolean
  note: string | null
}
export interface CatalogImage extends ImageItem {
  catalog: CatalogImageState
  values: Record<string, string | null>
  confidence: number | null
  values_source: 'review' | 'model' | 'none'
  model: { model_id: string; mode: 'live' | 'recorded'; created_at: string; recorded: RecordedInfo | null } | null
  prediction: PredictionResult | null
  review: Review | null
}
export interface CatalogProduct {
  product_id: string
  title: string
  brand: string | null
  category: string | null
  subcategory: string | null
  product_type: string | null
  image_count: number
  published_count: number
  status: 'live' | 'draft' | 'empty'
  cover: CompactImage | null
  perspectives: string[]
  min_confidence: number | null
  needs_review: number
}
export interface CatalogStats {
  products: number
  live_products: number
  draft_products: number
  empty_products: number
  images: number
  published_images: number
  hidden_by_rules: number
  visible_images: number
  needs_review: number
}
export interface StorefrontFilters {
  hide_non_photos: boolean
  only_white_background: boolean
  hide_people: boolean
  hide_added_text: boolean
  min_confidence: number
}
export interface CatalogResponse {
  products: CatalogProduct[]
  stats: CatalogStats
  filters: StorefrontFilters
  categories: string[]
}
export interface CatalogProductDetail {
  product: CatalogProduct
  images: CatalogImage[]
  filters: StorefrontFilters
}
export interface StorefrontProductCard {
  product_id: string
  title: string
  brand: string | null
  category: string | null
  subcategory: string | null
  product_type: string | null
  image_count: number
  cover: CompactImage
  perspectives: string[]
  min_confidence: number | null
}
export interface StorefrontResponse {
  products: StorefrontProductCard[]
  total: number
  filters: StorefrontFilters
  categories: { name: string; count: number }[]
  stats: { published_images: number; hidden_by_rules: number; visible_images: number }
}
export interface StorefrontProductDetail {
  product: StorefrontProductCard
  images: CompactImage[]
  filters: StorefrontFilters
}
