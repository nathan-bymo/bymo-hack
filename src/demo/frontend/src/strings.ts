/** Un solo punto di verità per i testi dell'interfaccia (inglese: D.2.1). */
export const S = {
  badgeDemo: 'Demo — Team Bymo · Hack Apertus',
  badge: {
    held_out: 'held-out',
    phase1_internal: 'phase-1 internal set (coverage)',
    seen_in_training: 'used for training',
    na: 'n/a',
    reported_only: 'reported only',
    recomputed: 'recomputed',
    offline_script: 'recomputed',
    kaggle_user: 'Kaggle dev (public)',
    live: 'Live',
    parityUnverified: 'parity not verified',
  },
  seenTooltip: 'This model learned from these images, so it is scored on the Kaggle leaderboard instead.',
  referenceLine: "Organisers' reference (Qwen 3.5 9B FT, dev)",
  referenceNote: 'Different set (Kaggle dev): not directly comparable with coverage.',
  bannerCoverage:
    '"coverage" is an internal phase-1 test set enriched with rare classes. It is meant to compare methods, not to estimate the Kaggle leaderboard.',
  status: { ai_suggested: 'AI suggested', reviewed: 'Reviewed', corrected: 'Corrected' },
  confidence: {
    head_softmax: 'Head softmax probability',
    letter_logprob: "Answer-letter probability (after the model's own description)",
    sequence_prob: 'Probability of the generated value',
    constrained_scoring: 'Constrained scoring probability',
    notComparable: 'Confidence values from different methods are not comparable.',
  } as Record<string, string>,
  invalidOutput: 'Invalid model output',
  notApplicable: 'Not scored for this kind of image (official rule applied to the prediction)',
  contextMissing: 'No product context: this condition was not measured for the trained models.',
  neverDevFinal: 'Never upload Kaggle dev or final images.',
  inactive: 'Not active in this demo',
  unavailable: {
    key: 'Unavailable: LLM_API_KEY is not configured',
    gpu: 'Unavailable: no GPU worker connected',
  },
  backendDown: 'Backend unreachable (/api/health)',
} as const

export const DIMENSION_NAMES: Record<string, string> = {
  subject: 'Subject',
  background: 'Background',
  perspective: 'Perspective',
  individuals: 'Individuals',
  text_classification: 'Overlay',
  text_language: 'Language',
}
export const DIMENSIONS = Object.keys(DIMENSION_NAMES)
