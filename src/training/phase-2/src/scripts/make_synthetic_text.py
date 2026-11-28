"""Synthetic training images for the weakest classes: text overlays in rare languages (Dutch, Italian, other, French),
generic certification seals, and wordless icon stickers (text_classification "other").

Source images: training_rows only (never the coverage set, never prompt-test), with a human "none" for both text
dimensions, a subject whose text dimensions are scored, and no person in the photo. A digitally added banner or
badge is a text overlay by the guideline, so the labels change consistently:
  - language banner:  text_classification = promotional_text, text_language = the language of the words;
  - seal badge:       text_classification = certification_seal, text_language = the language of its words;
  - icon sticker:     text_classification = other, text_language = none (no words).
The other dimensions keep the labels of the source image. The phrases are written by us; the seals are generic
shapes, not real certification marks.

Output (git-ignored): data/synthetic/images/*.jpg and data/synthetic/synthetic.csv with the columns of train.csv
(image_path relative to data/, image_id "syn_*", product_id of the source so that product splits stay intact).

  python src/scripts/make_synthetic_text.py --per-language 250 --other 250 --french 100 --seals 200 --icons 150
"""

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from common import images, splits  # noqa: E402

FONTS = [f"/System/Library/Fonts/Supplemental/{name}" for name in (
    "Arial.ttf", "Arial Bold.ttf", "Arial Black.ttf", "Arial Narrow Bold.ttf", "Arial Rounded Bold.ttf",
    "Georgia Bold.ttf", "Tahoma Bold.ttf", "Verdana Bold.ttf", "Trebuchet MS Bold.ttf", "Impact.ttf")]

PHRASES = {
    "Dutch": [
        "Nieuw! Nu met 20% korting", "Gratis verzending vanaf 50 euro", "Duurzaam en milieuvriendelijk",
        "Uitstekende kwaliteit voor elke dag", "Makkelijk schoon te maken", "Geschikt voor het hele gezin",
        "Nu nog voordeliger", "Lang houdbaar en sterk", "Veilig voor kinderen", "Snel en eenvoudig te monteren",
        "Bestseller van het jaar", "Met twee jaar garantie", "Perfect voor onderweg", "Extra zacht materiaal",
        "Waterdicht en ademend", "Bespaar energie elke dag", "Ontworpen in Nederland", "Op = op, wees er snel bij",
        "Inclusief handige opbergtas", "Krachtig, stil en zuinig"],
    "Italian": [
        "Novità! Ora con il 20% di sconto", "Spedizione gratuita sopra i 50 euro", "Sostenibile e rispettoso dell'ambiente",
        "Qualità eccellente per ogni giorno", "Facile da pulire", "Adatto a tutta la famiglia", "Ancora più conveniente",
        "Resistente e duraturo", "Sicuro per i bambini", "Montaggio rapido e semplice", "Il più venduto dell'anno",
        "Con due anni di garanzia", "Perfetto da portare in viaggio", "Materiale extra morbido",
        "Impermeabile e traspirante", "Risparmia energia ogni giorno", "Progettato in Italia",
        "Offerta limitata, affrettati", "Borsa di trasporto inclusa", "Potente, silenzioso ed efficiente"],
    "other": [  # languages outside the five listed ones
        "Nowość! Teraz 20% taniej", "Darmowa dostawa od 200 zł", "Wysoka jakość na co dzień", "Łatwy w czyszczeniu",
        "¡Nuevo! Ahora con un 20% de descuento", "Envío gratis a partir de 50 euros", "Fácil de limpiar y muy resistente",
        "Ideal para toda la familia", "Novinka! Nyní o 20 % levněji", "Doprava zdarma nad 1000 Kč",
        "Snadné čištění a dlouhá životnost", "Novidade! Agora com 20% de desconto", "Entrega gratuita acima de 50 euros",
        "Fácil de limpar e muito durável", "Nyhet! Nu 20 % billigare", "Fri frakt över 500 kr",
        "Lätt att rengöra och mycket hållbar", "Újdonság! Most 20% kedvezménnyel", "Könnyen tisztítható és tartós",
        "Nyhed! Nu med 20 % rabat", "Gratis fragt over 500 kr", "Yeni! Şimdi %20 indirimli"],
    "French": [
        "Nouveau ! Maintenant 20 % moins cher", "Livraison gratuite dès 50 euros", "Facile à nettoyer",
        "Idéal pour toute la famille", "Qualité supérieure au quotidien", "Résistant et durable",
        "Garantie de deux ans incluse", "Conçu pour durer", "Sans danger pour les enfants", "Montage simple et rapide"],
}
ICONS = ["♻", "✓", "⚡", "☀", "❄", "♥", "★", "☂", "✈", "⚙", "♨", "☎", "⌛", "⚠", "♫", "✿"]


def drawable(symbols, path="/System/Library/Fonts/Supplemental/Arial Unicode.ttf"):
    """Only the symbols the font really has: a missing glyph is drawn as the same empty box as an unused code point."""
    typeface = ImageFont.truetype(path, 64)

    def picture(text):
        image = Image.new("L", (96, 96), 0)
        ImageDraw.Draw(image).text((48, 48), text, font=typeface, fill=255, anchor="mm")
        return image.tobytes()

    missing = picture("\uffff")
    return [symbol for symbol in symbols if picture(symbol) != missing]
SEALS = {  # (language, words on the seal); generic, no real certification mark
    "English": ["QUALITY TESTED", "PREMIUM QUALITY", "ECO FRIENDLY", "100% NATURAL", "BEST CHOICE 2026", "SAFETY APPROVED"],
    "German": ["GEPRÜFTE QUALITÄT", "TESTSIEGER 2026", "UMWELTFREUNDLICH", "100% NATÜRLICH", "SICHERHEIT GEPRÜFT"],
    "French": ["QUALITÉ CONTRÔLÉE", "ÉCOLOGIQUE", "100% NATUREL", "SÉCURITÉ TESTÉE"],
    "Italian": ["QUALITÀ CONTROLLATA", "ECOLOGICO", "100% NATURALE", "SICUREZZA TESTATA"],
    "Dutch": ["GETESTE KWALITEIT", "MILIEUVRIENDELIJK", "100% NATUURLIJK", "VEILIGHEID GETEST"],
}
TEXT_FREE_SUBJECTS = {"full_product", "packaging", "package_and_product", "detail_closeup", "accessories",
                      "multipack_trade_unit", "delivery_scope", "product_grouping", "multiview"}


def candidate_rows(train):
    rows = splits.training_rows(train)
    human_none = lambda d: (rows[f"{d}_source"] == "human") & (rows[d] == "none")
    keep = (human_none("text_classification") & human_none("text_language") & rows.subject.isin(TEXT_FREE_SUBJECTS)
            & (rows.individuals != "individual_present"))
    rows = rows[keep].reset_index(drop=True)

    def big_enough(path):  # a few images are tiny or thin strips: no room for an overlay
        with Image.open(ROOT / "data" / path) as image:
            return min(image.size) >= 256

    return rows[rows.image_path.map(big_enough)].reset_index(drop=True)


def font(rng, size):
    for _ in range(10):
        path = rng.choice(FONTS)
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def colours(rng):
    pairs = [((220, 30, 40), (255, 255, 255)), ((255, 200, 0), (20, 20, 20)), ((20, 90, 200), (255, 255, 255)),
             ((30, 150, 70), (255, 255, 255)), ((20, 20, 20), (255, 255, 255)), ((255, 255, 255), (200, 20, 30))]
    return rng.choice(pairs)


def wrap(draw, text, typeface, width):
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=typeface) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return lines + [line]


def banner(image, text, rng):
    """A coloured strip with the phrase at the top or bottom, or a tilted ribbon in a corner."""
    draw = ImageDraw.Draw(image)
    w, h = image.size
    background, foreground = colours(rng)
    size = max(14, int(min(w, h) * rng.uniform(0.045, 0.075)))
    typeface = font(rng, size)
    lines = wrap(draw, text, typeface, int(w * rng.uniform(0.55, 0.9)))
    line_height = int(size * 1.25)
    box_h = line_height * len(lines) + size
    box_w = max(int(draw.textlength(line, font=typeface)) for line in lines) + 2 * size
    x = rng.choice([int(w * 0.03), (w - box_w) // 2, w - box_w - int(w * 0.03)])
    y = rng.choice([int(h * 0.03), h - box_h - int(h * 0.03)])
    draw.rounded_rectangle([x, y, x + box_w, y + box_h], radius=rng.choice([0, size // 2]), fill=background)
    for i, line in enumerate(lines):
        draw.text((x + size, y + size // 2 + i * line_height), line, font=typeface, fill=foreground)
    return image


def seal(image, words, rng):
    """A round badge with a star and short words, in a corner."""
    w, h = image.size
    d = int(min(w, h) * rng.uniform(0.24, 0.32))
    badge = Image.new("RGBA", (d, d), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    background, foreground = colours(rng)
    draw.ellipse([0, 0, d - 1, d - 1], fill=background, outline=foreground, width=max(2, d // 30))
    draw.ellipse([d * 0.08, d * 0.08, d * 0.92, d * 0.92], outline=foreground, width=max(1, d // 60))
    draw.text((d / 2, d * 0.3), "★", font=ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
                                                             int(d * 0.2)), fill=foreground, anchor="mm")
    path = rng.choice([f for f in FONTS if Path(f).exists()])
    for size in range(max(10, int(d * 0.13)), 7, -1):  # the largest size whose lines all fit inside the ring
        typeface = ImageFont.truetype(path, size)
        lines = wrap(draw, words, typeface, int(d * 0.68))
        if len(lines) <= 3 and all(draw.textlength(line, font=typeface) <= d * 0.72 for line in lines):
            break
    for i, line in enumerate(lines[:3]):
        draw.text((d / 2, d * 0.55 + i * d * 0.12), line, font=typeface, fill=foreground, anchor="mm")
    x = rng.choice([int(w * 0.03), w - d - int(w * 0.03)])
    y = rng.choice([int(h * 0.03), h - d - int(h * 0.03)])
    image.paste(badge, (x, y), badge)
    return image


def icons(image, rng):
    """One to three wordless pictogram stickers: an added element that is not text, a seal or a dimension."""
    w, h = image.size
    unicode_font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    for _ in range(rng.randint(1, 3)):
        d = int(min(w, h) * rng.uniform(0.13, 0.2))
        sticker = Image.new("RGBA", (d, d), (0, 0, 0, 0))
        draw = ImageDraw.Draw(sticker)
        background, foreground = colours(rng)
        if rng.random() < 0.5:
            draw.ellipse([0, 0, d - 1, d - 1], fill=background)
        else:
            draw.rounded_rectangle([0, 0, d - 1, d - 1], radius=d // 5, fill=background)
        draw.text((d / 2, d / 2), rng.choice(drawable(ICONS)), font=ImageFont.truetype(unicode_font, int(d * 0.6)),
                  fill=foreground, anchor="mm")
        x, y = rng.randint(int(w * 0.02), w - d - int(w * 0.02)), rng.choice([int(h * 0.03), h - d - int(h * 0.03)])
        image.paste(sticker, (x, y), sticker)
    return image


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--per-language", type=int, default=250, help="banners in Dutch and in Italian, each")
    parser.add_argument("--other", type=int, default=250, help="banners in other languages")
    parser.add_argument("--french", type=int, default=100, help="banners in French")
    parser.add_argument("--seals", type=int, default=200)
    parser.add_argument("--icons", type=int, default=150, help="wordless icon stickers (text_classification other)")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--out", type=Path, default=ROOT / "data" / "synthetic")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    train = pd.read_csv(ROOT / "data" / "train.csv", dtype=str, keep_default_na=False)
    pool = candidate_rows(train)
    jobs = [("Dutch", "banner")] * args.per_language + [("Italian", "banner")] * args.per_language
    jobs += [("other", "banner")] * args.other + [("French", "banner")] * args.french
    jobs += [(None, "seal")] * args.seals + [("none", "icons")] * args.icons
    if len(jobs) > len(pool):
        sys.exit(f"only {len(pool)} source images, {len(jobs)} requested")
    sources = pool.sample(n=len(jobs), random_state=args.seed).to_dict("records")
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for number, ((language, kind), source) in enumerate(zip(jobs, sources), 1):
        with Image.open(ROOT / "data" / source["image_path"]) as opened:
            image = images.opaque_rgb_on_white(opened)
        if max(image.size) > 1400:
            image.thumbnail((1400, 1400))
        if kind == "banner":
            image = banner(image, rng.choice(PHRASES[language]), rng)
            classification = "promotional_text"
        elif kind == "seal":
            language = rng.choice(list(SEALS))
            image = seal(image, rng.choice(SEALS[language]), rng)
            classification = "certification_seal"
        else:
            image = icons(image, rng)
            classification = "other"
        image_id = f"syn_{number:05d}"
        path = f"synthetic/images/{image_id}.jpg"
        image.save(ROOT / "data" / path, quality=92)
        row = dict(source)
        row.update({"image_id": image_id, "image_path": path,
                    "text_classification": classification, "text_classification_source": "human",
                    "text_classification_relevant": "true",
                    "text_language": language, "text_language_source": "human", "text_language_relevant": "true"})
        rows.append(row)
    frame = pd.DataFrame(rows, columns=train.columns)
    frame.to_csv(args.out / "synthetic.csv", index=False)
    print(f"{len(frame)} synthetic images from {len(pool)} candidates")
    print(frame.groupby(["text_classification", "text_language"]).size().to_string())


if __name__ == "__main__":
    main()
