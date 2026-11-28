/**
 * Nomi e descrizioni dei modelli: arrivano dal backend (`models.yaml`, in inglese) e si traducono dal testo inglese.
 * Questo elenco serve solo allo script di controllo, che così sa quali frasi devono esistere in ogni lingua.
 */
import { T } from './index'

export const MODEL_TEXTS = [
  T('Final model'), T('Our best. Apertus 8B fine-tuned together with an image encoder (SigLIP 2) and a text reader (OCR), on 9,645 labelled images.'),
  T('Ensemble of two models'), T('50/50 average of the SigLIP + VQ model and a DINOv3 model with 81 image tokens: our best public score.'),
  T('Apertus 8B, no training'), T('Apertus 8B as it comes. It first describes the image, then answers the six questions one by one.'),
  T('Apertus 70B, no training'), T('The large Apertus 70B as it comes, with the same questions as the 8B.'),
  T('LoRA + classifiers (P11)'), T('Apertus 8B with a light adapter (LoRA) and six small classifiers on top. 9,344 training images.'),
  T('Final + synthetic (P13)'), T('The final model trained one more epoch with 939 synthetic images for rare languages, seals and icons.'),
  T('Two image encoders (mixed)'), T("SigLIP 2 and DINOv3 image encoders with Apertus 8B, one decoder. The team's default delivery, 14 photos per second."),
  T('LoRA on the image encoder'), T('The final model with a LoRA adapter on SigLIP 2 too, so the image encoder learns the task. 10,683 images.'),
  T('Bigger image encoder (P14)'), T('The final model with a larger, sharper image encoder (SigLIP 2 so400m at 512²) and 196 image tokens.'),
  T('Final, all data (P15)'), T('The final model trained one more epoch on every labelled image, including our held-out test.'),
  T('LoRA + pooling, 512² (P12)'), T('P11 with sharper images (512²), left/right features and 2D attention pooling. 9,344 training images.'),
  T('Full fine-tune'), T('Every layer of Apertus 8B retrained, with six small classifiers on top. All 11,876 labelled images.'),
  T('This model does not answer in this demo.'),
  T('Galaxus × Apertus — Image classification'),   // titolo della scheda del browser (lo imposta I18nProvider)
]
