/** "What mattered": prove riportate dai file del team, ognuna con la sua fonte (file:riga). Nessun numero stimato qui. */

export interface Insight { title: string; body: string; source: string }

export const MATTERED: Insight[] = [
  {
    title: 'Left and right needed their own features',
    body: 'With only the average of the image tokens, the model picked the correct side 55% of the time: a coin flip, because a photo and its mirror look the same on average. Features that change sign when the image is mirrored, plus attention over the 2D image grid, took perspective from 0.544 to 0.755 on the held-out test, in the same step as more data and epochs.',
    source: 'src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:62, 84-87',
  },
  {
    title: 'Training Apertus beats prompting it',
    body: 'On the held-out test, Apertus 8B goes from 0.502 without training to 0.791 with a light adapter and six classifiers, and to 0.821 with the image encoder and OCR.',
    source: 'src/plain-benchmark/reports/apertus-8B-benchmark/prompt_eval_coverage.md:69; src/training/phase-1/reports/apertus-8B-lora-heads/results_full_h200.md:25; src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:62',
  },
  {
    title: 'Six short questions beat one long prompt',
    body: 'Without training, asking Apertus to describe the image and then answer one question at a time raised the rare-class test from 0.229 to 0.528 (8B) and from 0.304 to 0.583 (70B).',
    source: 'src/training/phase-1/reports/EXPERIMENTS.md:62-67',
  },
  {
    title: 'Reading the text helps the languages we have data for',
    body: 'Adding OCR text to the prompt lifted German from 0.79 to 0.86 and French from 0.63 to 0.69. Dutch, with a single training example, did not improve.',
    source: 'src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:93',
  },
  {
    title: 'Classifiers instead of generated text',
    body: 'Six classifier heads always return a valid class, in one pass. The adapter that wrote its answer as text gave 21 unusable answers out of 2,532.',
    source: 'src/training/phase-1/reports/apertus-8B-fullft-heads/results_phase2.md:44',
  },
]

export const DID_NOT_HELP: Insight[] = [
  {
    title: 'Retraining every layer',
    body: 'The full fine-tune scored 0.805 against 0.796 for the light adapter, with 27% more images: no clear gain, for an 80 GB GPU and 10.5 GB of weights instead of 0.3 GB.',
    source: 'src/training/phase-1/reports/apertus-8B-fullft-heads/results_phase2.md:32-34',
  },
  {
    title: 'Re-weighting the classes after training',
    body: 'Per-class corrections chosen on validation moved the rare-class test from 0.743 to 0.738 or 0.745: the imbalance was already handled during training.',
    source: 'src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:88',
  },
  {
    title: 'Random crops',
    body: 'A 10% crop cut into the product in 231 of 300 images, turning whole products into close-ups. We removed it.',
    source: 'src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:95-96',
  },
  {
    title: 'Mirrored second look at test time',
    body: '+0.019 on perspective, but −0.009 on people and 17% slower. Not adopted.',
    source: 'src/training/phase-2/reports/apertus-8B-vadapter-lora-heads/RISULTATI.md:97',
  },
]
