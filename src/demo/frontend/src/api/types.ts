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
  weights: { repo: string; revision: string; subfolder: string; size_gb: number } | null
  vram_estimate_gb: number | null
  headline_eval: EvalSummary | null
  kaggle_dev: { score: number | null; submitted_at?: string | null; note?: string }
}
export interface ModelsResponse {
  models: ModelSummary[]
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
}
export interface Health { status: string; version: string; git_commit: string | null; gpu_local: boolean; guard: { loaded: boolean; entries: number } }
