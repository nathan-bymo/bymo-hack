#!/usr/bin/env node
/**
 * Controllo delle traduzioni: ogni frase passata a tr(), t(), tn(), trn() o T() nel codice deve esistere in it.json, fr.json
 * e de.json, con gli stessi segnaposto {nome}. Segnala anche le frasi dei file che non servono più e le traduzioni
 * identiche all'inglese (a volte giuste, per esempio "Logo" o "Label").
 *
 *   node scripts/check-i18n.mjs            controlla (esce con 1 se manca qualcosa)
 *   node scripts/check-i18n.mjs --list     stampa tutte le frasi inglesi trovate nel codice
 */
import { readFileSync, readdirSync, statSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..', 'src')
const LANGS = ['it', 'fr', 'de']
// traduzioni uguali all'inglese perché giuste (nome proprio, parola uguale, sigla)
const SAME_OK = new Set(['Logo', 'A logo', 'Label', 'Live', 'live', 'Menu', 'Photo', 'Photos', 'Brand', 'Category', 'Subcategory', 'Product', 'Catalog',
  'Perspective', 'Perspectives', 'Date', 'Status', 'Cover', 'Admin view', 'No', 'EN', 'IT', 'FR', 'DE', 'n/a', 'Accessories', 'Description', 'Original', 'Test', 'Dutch', 'Question', '{n} photo', '{n} photos', 'position {n}',
  'LoRA + pooling, 512² (P12)', '(optional)', 'Unkategorisiert'])

function files(dir) {
  return readdirSync(dir).flatMap((name) => {
    const path = join(dir, name)
    return statSync(path).isDirectory() ? files(path) : /\.(ts|tsx)$/.test(name) ? [path] : []
  })
}
const unescape = (s) => s.replace(/\\(['"`\\])/g, '$1')
const LIT = String.raw`(?:'((?:\\.|[^'\\])*)'|"((?:\\.|[^"\\])*)"|` + '`' + String.raw`((?:\\.|[^` + '`' + String.raw`\\$])*)` + '`)'
const one = new RegExp(String.raw`(?<![\w.])(?:tr|t|T)\(\s*` + LIT, 'g')
const plural = new RegExp(String.raw`(?<![\w.])(?:tn|trn)\(\s*[^,()]+(?:\([^()]*\))?[^,()]*,\s*` + LIT + String.raw`\s*,\s*` + LIT, 'g')
const pick = (m, i) => unescape(m[i] ?? m[i + 1] ?? m[i + 2] ?? '')

const found = new Map()   // frase → file
for (const file of files(root)) {
  if (file.includes('/i18n/index.tsx')) continue
  const text = readFileSync(file, 'utf8')
  for (const m of text.matchAll(one)) found.set(pick(m, 1), file)
  for (const m of text.matchAll(plural)) { found.set(pick(m, 1), file); found.set(pick(m, 4), file) }
}
if (process.argv.includes('--list')) {
  console.log(JSON.stringify([...found.keys()].sort(), null, 1))
  process.exit(0)
}

const holes = (s) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort().join(',')
let problems = 0
for (const lang of LANGS) {
  const dict = JSON.parse(readFileSync(join(root, 'i18n', `${lang}.json`), 'utf8'))
  const missing = [], broken = [], same = [], unused = [], empty = []
  for (const key of found.keys()) {
    if (!(key in dict)) { missing.push(key); continue }
    if (!dict[key].trim()) empty.push(key)
    if (holes(key) !== holes(dict[key])) broken.push(key)
    if (dict[key] === key && !SAME_OK.has(key) && /[A-Za-z]{4,}/.test(key)) same.push(key)
  }
  for (const key of Object.keys(dict)) if (!found.has(key)) unused.push(key)
  const show = (label, list) => { if (list.length) { console.log(`\n[${lang}] ${label} (${list.length})`); list.forEach((k) => console.log('  -', k)) } }
  show('MANCANTI', missing); show('VUOTE', empty); show('SEGNAPOSTO DIVERSI', broken); show('UGUALI ALL\'INGLESE (da controllare)', same); show('NON USATE', unused)
  problems += missing.length + broken.length + empty.length
}
console.log(`\n${found.size} frasi nel codice, ${LANGS.length} lingue: ${problems ? problems + ' problemi' : 'tutto presente'}`)
process.exit(problems ? 1 : 0)
