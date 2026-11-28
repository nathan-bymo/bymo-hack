/** Tutte le prove del team, in ordine (src/training/EXPERIMENTS.md). Ogni numero vale solo sul suo set. */

export type SetKey = 'kaggle' | 'heldout' | 'rare' | 'dev600' | 'balanced450'

export const SET_NAME: Record<SetKey, string> = {
  kaggle: 'Kaggle leaderboard',
  heldout: 'held-out test, 1,193 photos',
  rare: 'rare-class test, 2,532 photos',
  dev600: 'first 600 prompt-dev photos',
  balanced450: 'balanced test, 450 photos',
}

export interface Result { score: string; set: SetKey; label?: string }
export interface Experiment { id: string; title: string; what: string; data: string; results: Result[]; note?: string; source: string; final?: boolean }
export interface Phase { title: string; lesson: string; items: Experiment[] }

const E = 'src/training/EXPERIMENTS.md'
const B = 'src/training/benchmark/overall.md'

export const PHASES: Phase[] = [
  {
    title: 'Ask Apertus as it is',
    lesson: 'Without training, the way we ask matters a lot, but even the best prompt stays far from the target.',
    items: [
      { id: 'P0', title: 'What the Apertus API can do', what: 'Apertus 8B and 70B through the Hack Apertus API: image input, answer probabilities, speed.', data: 'no training', results: [], note: 'Feasibility check on a test image, no score.', source: `${E}:28` },
      { id: 'P1', title: 'One long prompt with every rule', what: 'Apertus 8B, four prompt variants with the full labelling rules.', data: 'no training', results: [{ score: '0.327', set: 'dev600', label: 'best of 4' }], source: `${B}:93` },
      { id: 'P2', title: 'Why the long prompt gets the subject wrong', what: '15 configurations of 8B and 70B on the subject question only.', data: 'no training', results: [], note: 'On 103 photos, subject F1 for 8B went from 0.01 with the long rules to 0.54 when it first describes the photo, without the rules.', source: 'src/plain-benchmark/apertus-8B-benchmark/prompt_eval.md:88-104' },
      { id: 'P3', title: 'Six short questions instead of one', what: 'Apertus describes the photo, then answers one question at a time.', data: 'no training', results: [{ score: '0.482', set: 'dev600', label: '8B' }, { score: '0.454', set: 'dev600', label: '70B' }], source: `${B}:98, 81` },
      { id: 'P4', title: 'The best prompt on the big test', what: 'Five prompt variants on the rare-class test.', data: 'no training', results: [{ score: '0.229 → 0.528', set: 'rare', label: '8B' }, { score: '0.304 → 0.583', set: 'rare', label: '70B' }], source: `${E}:64-69` },
      { id: 'P5', title: 'Can 70B teach 8B?', what: 'Analysis of the 70B answers as training targets.', data: 'no training', results: [], note: 'Analysis only, no new model.', source: `${E}:33` },
    ],
  },
  {
    title: 'Small classifiers on a frozen Apertus',
    lesson: 'Apertus already knows a lot about the photos: six small classifiers on its frozen features double the pixel baseline.',
    items: [
      { id: 'P6', title: 'Pixels only, as a floor', what: '24×24 pixels and six classifiers, no Apertus.', data: '8,471 photos', results: [{ score: '0.366', set: 'rare' }], source: `${E}:67` },
      { id: 'P7', title: 'The same floor on the small split', what: '24×24 pixels and six classifiers, four losses.', data: '1,199 photos', results: [{ score: '0.257', set: 'balanced450' }], source: `${E}:51` },
      { id: 'P8', title: 'Classifiers on a frozen Apertus 8B', what: 'Apertus 8B unchanged; only six classifiers are trained. Four losses.', data: '1,199 photos', results: [{ score: '0.542', set: 'balanced450', label: 'best of 4' }], source: `${E}:55` },
      { id: 'P9', title: 'Class weights and focal loss', what: 'Six loss variants, five seeds each.', data: '1,199 photos', results: [{ score: '0.557', set: 'balanced450', label: 'strong class weights, mean of 5 seeds' }], note: 'Focal loss did not beat cross-entropy.', source: `${E}:56-57` },
    ],
  },
  {
    title: 'Teach Apertus itself',
    lesson: 'Adapting Apertus with LoRA beats every prompt and every frozen setup; retraining all layers adds little.',
    items: [
      { id: 'P10', title: 'A light adapter, answers as text', what: 'Apertus 8B with LoRA, writing its answers as JSON.', data: '1,500 photos', results: [{ score: '0.696', set: 'balanced450' }, { score: '0.734', set: 'rare' }], source: `${E}:58, 70` },
      { id: 'P11', title: 'Adapter and six classifiers, all the data', what: 'Apertus 8B with LoRA and six classifiers on top, 256² images.', data: '9,344 photos', results: [{ score: '0.796', set: 'kaggle' }, { score: '0.791', set: 'heldout' }, { score: '0.781', set: 'rare' }], source: `${E}:71, 80` },
      { id: 'FT', title: 'Retrain every layer', what: 'Full fine-tune of Apertus 8B with six classifiers.', data: '11,876 photos', results: [{ score: '0.805', set: 'kaggle' }], note: 'About the same as P11, for an 80 GB GPU and 10.5 GB of weights.', source: `${E}:83` },
    ],
  },
  {
    title: 'Give Apertus better eyes',
    lesson: 'Seeing where things are is what moved the score: perspective went from a coin flip to 0.755.',
    items: [
      { id: 'Final', title: 'Image encoder, OCR and left/right features', what: 'SigLIP 2 image tokens and docTR text added to Apertus with LoRA; features that flip with the mirror image and 2D attention pooling.', data: '9,645 photos', results: [{ score: '0.832', set: 'kaggle' }, { score: '0.821', set: 'heldout' }], source: `${E}:84`, final: true },
      { id: 'P12', title: 'Left/right features on P11, sharper images', what: 'P11 with 512² images, the same left/right features and attention pooling.', data: '9,344 photos', results: [{ score: '0.806', set: 'kaggle' }, { score: '0.791', set: 'heldout' }, { score: '0.806', set: 'rare' }], source: `${E}:82` },
      { id: 'P13', title: 'Synthetic photos for the rare classes', what: 'The final model, one more epoch with 939 generated photos for rare languages, seals and icons.', data: '9,645 + 939 photos', results: [{ score: '0.829', set: 'kaggle' }, { score: '0.847', set: 'heldout' }], note: 'Better on our test, not on Kaggle: the final submission stays the final model.', source: `${E}:81` },
      { id: 'P14', title: 'A bigger, sharper image encoder', what: 'The final model with SigLIP 2 so400m at 512² and 196 image tokens instead of 49.', data: '9,645 photos', results: [{ score: '0.833', set: 'heldout' }], note: 'Not submitted to Kaggle.', source: `${E}:42; ${B}:17` },
      { id: 'P15', title: 'The final model on every labelled photo', what: 'One more epoch of the final model on all 11,876 labelled photos, our held-out test included.', data: '11,876 photos', results: [{ score: '0.831', set: 'kaggle' }], note: 'Level with the final model (0.832): more data did not help.', source: `${E}:83` },
      { id: 'Mixed', title: 'Two image encoders', what: 'DINOv3 features added to the SigLIP 2 path of the final model, plus synthetic text overlays; one decoder. With a faster inference path and TF32 it reads 14.3 photos per second on one H100.', data: '10,683 + 1,040 photos', results: [{ score: '0.842', set: 'kaggle' }, { score: '0.843', set: 'heldout' }], note: 'The team’s default delivery: best single model on Kaggle.', source: 'src/training/apertus-8B-mixed-lora-heads/RESULTS.md:24-25, 54, 142' },
      { id: 'SigLIP LoRA', title: 'Teach the image encoder too', what: 'The final model with a LoRA adapter on SigLIP 2, so the image encoder is trained as well.', data: '10,683 photos', results: [{ score: '0.835', set: 'kaggle' }, { score: '0.836', set: 'heldout', label: 'checkpoint chosen on this test' }], source: 'src/training/apertus-8B-vadapter-lora-heads/results_siglip_lora.md:13-14; src/training/benchmark/README.md:28' },
      { id: 'Ensemble', title: 'Two models averaged', what: 'Average of the probabilities of the SigLIP + VQ model and a DINOv3 model with 81 image tokens.', data: 'two trained models', results: [{ score: '0.847', set: 'kaggle' }], note: 'Our best score, but two decoders halve the speed: used only as a teacher.', source: 'src/training/benchmark/README.md:26; src/training/apertus-8B-mixed-lora-heads/RESULTS.md:53' },
    ],
  },
]
