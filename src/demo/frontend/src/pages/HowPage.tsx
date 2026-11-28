import type { ReactNode } from 'react'
import { AppShell } from '../components/app'
import { PHASES, SET_NAME, type Experiment } from '../lib/experiments'
import { DID_NOT_HELP, MATTERED, type Insight } from '../lib/insights'

const SRC = 'src/training/apertus-8B-vadapter-lora-heads/RISULTATI.md'

function InsightList({ items, testId }: { items: Insight[]; testId: string }) {
  return (
    <ul className="m-0 mt-5 grid list-none gap-x-12 gap-y-6 p-0 md:grid-cols-2" data-testid={testId}>
      {items.map((i) => (
        <li key={i.title} className="border-t border-graphite pt-3">
          <h3 className="m-0 text-[17px] font-bold leading-snug tracking-tight">{i.title}</h3>
          <p className="m-0 mt-1.5 text-[15px] leading-relaxed">{i.body}</p>
          <p className="m-0 mt-1.5 text-[12px] text-slate">Source: {i.source}</p>
        </li>
      ))}
    </ul>
  )
}

/** Un riquadro dello schema: pieno = allenato da noi, contorno = modello pubblicato, usato così com'è. */
function Box({ title, body, trained }: { title: string; body?: string; trained?: boolean }) {
  return (
    <div className={`px-3 py-2.5 ${trained ? 'bg-graphite text-white' : 'border border-graphite bg-white'}`}>
      <div className="text-[15px] font-bold leading-snug">{title}</div>
      {body && <div className={`mt-0.5 text-[13px] leading-snug ${trained ? 'text-white/80' : 'text-slate'}`}>{body}</div>}
    </div>
  )
}

function Stage({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="flex flex-col gap-2">
      <div className="text-[13px] text-slate">{label}</div>
      {children}
    </div>
  )
}

const Arrow = () => <div aria-hidden="true" className="hidden self-center pt-6 text-[22px] text-slate lg:block">→</div>

function Pipeline() {
  return (
    <figure className="m-0 mt-8" data-testid="pipeline">
      <div className="grid gap-4 lg:grid-cols-[minmax(0,0.8fr)_auto_minmax(0,1.3fr)_auto_minmax(0,1.1fr)_auto_minmax(0,1fr)_auto_minmax(0,0.9fr)]">
        <Stage label="Input">
          <Box title="Product photo" body="JPEG, PNG or WebP, any size" />
          <Box title="Product details" body="Title, type, brand, category. Optional" />
        </Stage>
        <Arrow />
        <Stage label="Three ways to read the photo">
          <div>
            <Box title="SigLIP 2 image encoder" body="Image features of the photo, frozen" />
            <div className="ml-5 border-l-2 border-graphite pl-0">
              <Box title="Projector" body="Turns them into 49 tokens Apertus can read" trained />
            </div>
          </div>
          <Box title="Apertus image tokenizer" body="About 256 image tokens" />
          <Box title="docTR text reader (OCR)" body="The words printed on the image" />
        </Stage>
        <Arrow />
        <Stage label="Language model">
          <Box title="Apertus v1.5 8B" body="24 of its 32 layers" />
          <Box title="LoRA adapter" body="Rank 32, trained on 9,645 labelled photos" trained />
        </Stage>
        <Arrow />
        <Stage label="Where things are">
          <Box title="Left and right features" body="They change sign when the photo is mirrored" trained />
          <Box title="2D attention pooling" body="Looks at each image token in its place on the grid" trained />
        </Stage>
        <Arrow />
        <Stage label="Output">
          <Box title="Six classifiers" body="One per question, all in one pass" trained />
          <div className="text-[14px] leading-relaxed">Subject, background, perspective, people, text overlay, language: one class each, with its probability.</div>
        </Stage>
      </div>
      <figcaption className="mt-4 flex flex-wrap items-center gap-x-6 gap-y-2 text-[13px] text-slate">
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-5 bg-graphite" /> Trained by us: 0.37 GB of weights</span>
        <span className="flex items-center gap-2"><span className="inline-block h-3 w-5 border border-graphite" /> Open-weights models, used as published</span>
        <span>Source: {SRC}:20-34</span>
      </figcaption>
    </figure>
  )
}

const WALK = [
  { title: 'Read the photo three ways', body: 'SigLIP 2 turns it into 49 image tokens, Apertus’ own image tokenizer into about 256 more, and docTR reads the words printed on it.' },
  { title: 'Apertus reads everything', body: 'Apertus 8B takes the tokens, the printed words and the product details when there are any. A LoRA adapter teaches it the task; with the classifiers, everything we trained weighs 0.37 GB.' },
  { title: 'Work out where things are', body: 'Features that change sign when the photo is mirrored, and attention over the 2D grid of image tokens, let it tell left from right.' },
  { title: 'Answer the six questions', body: 'Six small classifiers give one class each, with its probability, in a single pass: 4.9 photos per second on one L40S GPU.' },
]

function ExperimentRow({ e }: { e: Experiment }) {
  return (
    <li data-testid="experiment" className={`grid gap-x-6 gap-y-1 border-b border-rule py-3 md:grid-cols-[4.5rem_minmax(0,1fr)_17rem] ${e.final ? 'border-l-4 border-l-graphite pl-3' : ''}`}>
      <span className="text-[15px] font-bold">{e.id}</span>
      <span>
        <span className="block text-[16px] font-bold leading-snug">{e.title}</span>
        <span className="block text-[14px] leading-snug text-slate">{e.what} Training data: {e.data}.</span>
        {e.note && <span className="mt-0.5 block text-[14px] leading-snug">{e.note}</span>}
        <span className="mt-0.5 block text-[11px] text-slate">Source: {e.source}</span>
      </span>
      <span className="space-y-1">
        {e.results.map((r) => (
          <span key={r.set + r.score + (r.label ?? '')} className="flex items-baseline justify-between gap-3">
            <span className="text-[13px] leading-tight text-slate">{SET_NAME[r.set]}{r.label ? `, ${r.label}` : ''}</span>
            <span className={`shrink-0 text-[17px] tabular-nums ${r.set === 'kaggle' ? 'font-bold' : ''}`}>{r.score}</span>
          </span>
        ))}
      </span>
    </li>
  )
}

const STEPS = [
  { what: 'Adapter, projector and six classifiers, trained from scratch', data: '8,403 photos, 2 epochs', time: '31 min', cost: '$4.80', macro: '0.750', persp: '0.586' },
  { what: 'Added the OCR text to the prompt, removed random crops', data: 'same photos, +1 epoch', time: '19 min', cost: '$2.70', macro: '0.741', persp: '0.544' },
  { what: 'Added left/right features and 2D attention pooling', data: '9,645 photos, +3 epochs', time: '63 min', cost: '$6.10', macro: '0.821', persp: '0.755' },
]

export default function HowPage() {
  return (
    <AppShell>
      <h1>How it works</h1>
      <p className="lead mt-3">
        Galaxus needs six answers for every product photo. We taught Apertus 8B to give them: it reads the photo three ways,
        and six small classifiers answer the six questions in one pass.
      </p>

      <section className="mt-12">
        <h2>The final model</h2>
        <Pipeline />
        <ol className="m-0 mt-10 grid list-none gap-x-10 gap-y-6 p-0 md:grid-cols-2 lg:grid-cols-4" data-testid="walkthrough">
          {WALK.map((w, i) => (
            <li key={w.title} className="border-t-2 border-graphite pt-3">
              <span className="block text-[13px] text-slate">Step {i + 1}</span>
              <h3 className="m-0 mt-0.5 text-[17px] font-bold leading-snug tracking-tight">{w.title}</h3>
              <p className="m-0 mt-1.5 text-[15px] leading-relaxed">{w.body}</p>
            </li>
          ))}
        </ol>
      </section>

      <section className="mt-16">
        <h2>Training the final model</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Three training runs, each starting from the one before, on one H200 GPU. Scores are on our held-out test of 1,193 photos.
          About $16 of GPU in total, including one run that ran out of memory.
        </p>
        <div className="mt-6 overflow-x-auto">
          <table className="w-full border-collapse text-left" data-testid="training-steps">
            <thead>
              <tr className="border-b-2 border-graphite text-[13px] text-slate">
                <th className="py-2 pr-4 font-normal">Step</th>
                <th className="py-2 pr-4 font-normal">What changed</th>
                <th className="py-2 pr-4 font-normal">Data</th>
                <th className="py-2 pr-4 text-right font-normal">Training</th>
                <th className="py-2 pr-4 text-right font-normal">Cost</th>
                <th className="py-2 pr-4 text-right font-normal">Perspective</th>
                <th className="py-2 text-right font-normal">Macro F1</th>
              </tr>
            </thead>
            <tbody>
              {STEPS.map((s, i) => (
                <tr key={s.what} className="border-b border-rule align-baseline">
                  <td className="py-3 pr-4 text-[16px] font-bold">{i + 1}</td>
                  <td className="py-3 pr-4 text-[16px]">{s.what}</td>
                  <td className="py-3 pr-4 text-[15px] text-slate">{s.data}</td>
                  <td className="py-3 pr-4 text-right text-[15px] tabular-nums">{s.time}</td>
                  <td className="py-3 pr-4 text-right text-[15px] tabular-nums">{s.cost}</td>
                  <td className="py-3 pr-4 text-right text-[18px] tabular-nums">{s.persp}</td>
                  <td className={`py-3 text-right text-[18px] tabular-nums ${i === STEPS.length - 1 ? 'font-bold' : ''}`}>{s.macro}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="m-0 mt-2 text-[12px] text-slate">Source: {SRC}:45-47 and 60-62. Step 3 changed data, layers and epochs together, so its gain cannot be split between them.</p>
      </section>

      <section className="mt-16" data-testid="experiments">
        <h2>Every experiment we ran</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          From prompting Apertus as it comes to the final model, in the order we ran them. Each score is on the test set named
          next to it, and only scores on the same set can be compared. Bold scores are from the Kaggle leaderboard.
        </p>
        {PHASES.map((ph, i) => (
          <div key={ph.title} className="mt-10">
            <h3 className="m-0 text-[20px] font-bold tracking-tight">{i + 1}. {ph.title}</h3>
            <p className="m-0 mt-1 max-w-[75ch] text-[15px]">{ph.lesson}</p>
            <ol className="m-0 mt-3 list-none border-t border-graphite p-0">
              {ph.items.map((e) => <ExperimentRow key={e.id} e={e} />)}
            </ol>
          </div>
        ))}
      </section>

      <section className="mt-16">
        <h2>What mattered</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">The changes that moved the score most, measured on the same images before and after.</p>
        <InsightList items={MATTERED} testId="insights-mattered" />
      </section>

      <section className="mt-14">
        <h2>What did not help</h2>
        <InsightList items={DID_NOT_HELP} testId="insights-not" />
      </section>

      <section className="mt-16" data-testid="runtime-models">
        <h2>What runs at inference</h2>
        <p className="m-0 mt-1 max-w-[70ch] text-[15px] text-slate">
          Only Apertus and open-weights helper models: no closed model and no commercial API. The trained models run on a
          single GPU, so they can be installed on-premises, on an air-gapped server or in a Swiss cloud.
        </p>
        <table className="mt-5 w-full border-collapse text-left">
          <thead>
            <tr className="border-b-2 border-graphite text-[13px] text-slate">
              <th className="py-2 pr-4 font-normal">Model</th><th className="py-2 pr-4 font-normal">Role</th><th className="py-2 font-normal">Used by</th>
            </tr>
          </thead>
          <tbody className="text-[15px]">
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">Apertus v1.5 8B</td><td className="py-2.5 pr-4">Reads the image tokens, the OCR text and the product details; the six classifiers sit on top</td><td className="py-2.5 text-slate">Final model, P13, P12, P11 and the full fine-tune on our GPU; 8B without training through the Hack Apertus inference API</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">Apertus v1.5 70B</td><td className="py-2.5 pr-4">Describes the image and answers the six questions</td><td className="py-2.5 text-slate">70B without training, through the Hack Apertus inference API</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">SigLIP 2 base</td><td className="py-2.5 pr-4">Image encoder that feeds Apertus 49 extra image tokens</td><td className="py-2.5 text-slate">Final model, P13</td></tr>
            <tr className="border-b border-rule align-baseline"><td className="py-2.5 pr-4 font-bold">docTR</td><td className="py-2.5 pr-4">Reads the text printed on the image (OCR)</td><td className="py-2.5 text-slate">Final model, P13</td></tr>
          </tbody>
        </table>
        <p className="m-0 mt-2 text-[12px] text-slate">Sources: {SRC}:119-120 (google/siglip2-base-patch16-224; docTR fast_base + Felix92/doctr-torch-parseq-multilingual-v1).</p>
      </section>
    </AppShell>
  )
}
