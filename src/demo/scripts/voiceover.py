#!/usr/bin/env python3
"""Voce fuori campo del video con ElevenLabs, dalle battute di docs/demo/SCRIPT_VIDEO.md (righe che iniziano con "> ").

    python src/demo/scripts/voiceover.py --list-voices            # elenca le voci del proprio account
    python src/demo/scripts/voiceover.py --voice "<nome o id>"     # un MP3 per scena + uno completo

La chiave si legge da ELEVENLABS_API_KEY (ambiente o .env nella radice del repo) e non viene mai stampata.
Il testo inviato a ElevenLabs è solo quello dello script del video: nessun dato della competizione.
Gli MP3 vanno in ~/Desktop/Bymo demo - voce/ (fuori dal repo).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
SCRIPT = REPO / "docs" / "demo" / "SCRIPT_VIDEO.md"
API = "https://api.elevenlabs.io/v1"


def api_key() -> str:
    key = os.environ.get("ELEVENLABS_API_KEY")
    env = REPO / ".env"
    if not key and env.exists():
        for line in env.read_text().splitlines():
            if line.strip().startswith("ELEVENLABS_API_KEY="):
                key = line.split("=", 1)[1].strip().strip('"').strip("'")
    if not key:
        sys.exit("Manca ELEVENLABS_API_KEY: aggiungila a .env (ELEVENLABS_API_KEY=...) e riprova.")
    return key


def request(path: str, key: str, body: dict | None = None) -> bytes:
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{API}{path}", data=data, method="POST" if body is not None else "GET",
                                 headers={"xi-api-key": key, "Content-Type": "application/json",
                                          "Accept": "audio/mpeg" if body is not None else "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.read()
    except urllib.error.HTTPError as err:  # il messaggio di ElevenLabs non contiene la chiave
        sys.exit(f"ElevenLabs ha risposto {err.code}: {err.read().decode(errors='replace')[:300]}")


def scenes() -> list[tuple[str, str]]:
    """(titolo della scena, testo da leggere) nell'ordine dello script."""
    out, title, lines = [], None, []
    for line in SCRIPT.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^## (Scena \d+|Chiusura)(.*)$", line)
        if heading or line.startswith("## "):
            if title and lines:
                out.append((title, " ".join(lines)))
            title, lines = (heading.group(1) if heading else None), []
        elif title and line.startswith("> "):
            lines.append(line[2:].strip())
    if title and lines:
        out.append((title, " ".join(lines)))
    return out


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--list-voices", action="store_true")
    parser.add_argument("--voice", help="nome o voice_id di una voce dell'account")
    parser.add_argument("--model", default="eleven_multilingual_v2")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--out", type=Path, default=Path.home() / "Desktop" / "Bymo demo - voce")
    parser.add_argument("--dry", action="store_true", help="mostra solo le battute, senza chiamare ElevenLabs")
    args = parser.parse_args(argv)

    parts = scenes()
    if args.dry:
        for title, text in parts:
            print(f"{title} ({len(text.split())} parole): {text}\n")
        print(f"Totale: {sum(len(t.split()) for _, t in parts)} parole")
        return
    key = api_key()
    voices = json.loads(request("/voices", key))["voices"]
    if args.list_voices or not args.voice:
        for v in voices:
            labels = ", ".join(f"{k}: {val}" for k, val in (v.get("labels") or {}).items())
            print(f"{v['name']:<28} {v['voice_id']}  {labels}")
        if not args.voice:
            print("\nScegli una voce e rilancia con --voice \"<nome o id>\".")
        return
    match = [v for v in voices if args.voice in (v["voice_id"], v["name"])]
    if not match:
        sys.exit(f"Voce '{args.voice}' non trovata: usa --list-voices.")
    voice_id = match[0]["voice_id"]
    args.out.mkdir(parents=True, exist_ok=True)
    settings = {"stability": args.stability, "similarity_boost": 0.75}
    for i, (title, text) in enumerate(parts, 1):
        audio = request(f"/text-to-speech/{voice_id}", key, {"text": text, "model_id": args.model, "voice_settings": settings})
        path = args.out / f"{i:02d} {title}.mp3"
        path.write_bytes(audio)
        print(f"{path.name}: {len(text.split())} parole")
    full = " ".join(text for _, text in parts)
    (args.out / "00 completo.mp3").write_bytes(
        request(f"/text-to-speech/{voice_id}", key, {"text": full, "model_id": args.model, "voice_settings": settings}))
    print(f"Fatto: {args.out}")


if __name__ == "__main__":
    main()
