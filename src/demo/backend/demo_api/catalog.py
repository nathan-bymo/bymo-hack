"""Catalogo della demo: prodotti, foto, pubblicazione in vetrina e regole del negozio.

Il catalogo non è un secondo dataset: è una vista sopra `images`, `predictions` e `reviews`, più due tabelle
(`products`, `product_images`) che tengono le decisioni dell'admin. La vetrina legge solo le foto pubblicate e
applica i filtri scelti dall'admin, così il lato cliente cambia davvero quando cambia la regola.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from . import predictions as P
from .db import Database, now_iso
from .errors import ApiError
from .relevance import applicable
from .routers.images import image_dict
from .routers.reviews import review_dict

DIMENSIONS = ("subject", "background", "perspective", "individuals", "text_classification", "text_language")

# Ordine di preferenza quando la stessa foto ha risposte di più modelli: prima il modello finale, poi gli altri
# modelli allenati, infine lo zero-shot. La vetrina mostra la risposta più affidabile disponibile.
PREFERRED_MODELS = (
    "apertus-8b-vadapter-lora-heads-final",
    "apertus-8b-mixed-lora-heads",
    "apertus-8b-siglip-lora",
    "apertus-8b-vadapter-lora-heads-p15",
    "apertus-8b-vadapter-lora-heads-p13",
    "apertus-8b-lora-heads-p11",
    "apertus-8b-fullft-heads",
    "apertus-8b-lora-heads-p12",
    "apertus-8b-vadapter-lora-heads-p14",
    "apertus-70b-zeroshot-v2",
    "apertus-8b-zeroshot-v2",
)

# Regola unica per l'ordine delle viste di un prodotto, valida in catalogo, vetrina e per la copertina automatica:
#   1. viste ad angolo a sinistra, 2. viste ad angolo a destra (davanti prima del retro),
#   3. tutte le altre viste (frontale, lati, retro, dall'alto, dal basso),
#   4. in fondo le foto senza vista letta. A parità di vista resta l'ordine di inserimento.
# La copertina scelta dall'admin va sempre per prima; se non ne ha scelta una, lo diventa la prima vista in regola.
PERSPECTIVE_ORDER = (
    "front_angle_left", "back_angle_left", "front_angle_right", "back_angle_right",
    "front_head_on", "left_side", "right_side", "back_head_on", "top_down", "bottom_up",
)

NON_PHOTO_SUBJECTS = {"placeholder", "logo", "energy_label", "infographic"}

DEFAULT_FILTERS: dict[str, Any] = {
    "hide_non_photos": True,
    "only_white_background": False,
    "hide_people": False,
    "hide_added_text": False,
    "min_confidence": 0.0,
}

FILTERS_KEY = "storefront_filters"


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _product_id(title: str, brand: str | None) -> str:
    raw = f"{title.strip().lower()}|{(brand or '').strip().lower()}"
    return "prd_" + hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def ensure_product(db: Database, title: str, brand: str | None = None, category: str | None = None,
                   subcategory: str | None = None, product_type: str | None = None) -> tuple[dict, bool]:
    """Restituisce (prodotto, creato). Stesso titolo e marca = stesso prodotto."""
    product_id = _product_id(title, brand)
    row = db.one("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if row:
        return row, False
    now = now_iso()
    db.execute(
        "INSERT INTO products (product_id, title, brand, category, subcategory, product_type, created_at, updated_at) "
        "VALUES (?,?,?,?,?,?,?,?)",
        (product_id, title.strip(), _clean(brand), _clean(category), _clean(subcategory), _clean(product_type), now, now),
    )
    return db.one("SELECT * FROM products WHERE product_id = ?", (product_id,)), True  # type: ignore[return-value]


def _latest_predictions(db: Database, image_ids: list[str]) -> dict[str, dict]:
    if not image_ids:
        return {}
    marks = ",".join("?" for _ in image_ids)
    rows = db.all(
        f"SELECT * FROM predictions WHERE image_id IN ({marks}) AND valid = 1 "
        "ORDER BY created_at DESC, rowid DESC", image_ids)
    best: dict[str, dict] = {}
    for row in rows:
        current = best.get(row["image_id"])
        if current is None:
            best[row["image_id"]] = row
            continue
        rank = PREFERRED_MODELS.index(row["model_id"]) if row["model_id"] in PREFERRED_MODELS else len(PREFERRED_MODELS)
        current_rank = (PREFERRED_MODELS.index(current["model_id"]) if current["model_id"] in PREFERRED_MODELS
                        else len(PREFERRED_MODELS))
        if rank < current_rank:
            best[row["image_id"]] = row
    return best


def _latest_reviews(db: Database, image_ids: list[str]) -> dict[str, dict]:
    if not image_ids:
        return {}
    marks = ",".join("?" for _ in image_ids)
    rows = db.all(
        f"SELECT * FROM reviews WHERE image_id IN ({marks}) ORDER BY created_at DESC, rowid DESC", image_ids)
    out: dict[str, dict] = {}
    for row in rows:
        out.setdefault(row["image_id"], review_dict(row))
    return out


def _values_and_confidence(pred: dict | None, review: dict | None) -> tuple[dict[str, str | None], float | None, str]:
    confs: dict[str, float | None] = {d: None for d in DIMENSIONS}
    if review:
        values = {d: review["values"].get(d) for d in DIMENSIONS}
        source = "review"
    elif pred:
        dims = json.loads(pred["dimensions_json"])
        values = {d: dims.get(d, {}).get("label") for d in DIMENSIONS}
        source = "model"
    else:
        values = {d: None for d in DIMENSIONS}
        source = "none"
    if pred:
        dims = json.loads(pred["dimensions_json"])
        confs = {d: dims.get(d, {}).get("confidence") for d in DIMENSIONS}
    ok = applicable(values)
    usable = [confs[d] for d in DIMENSIONS if ok.get(d) and confs.get(d) is not None]
    return values, (min(usable) if usable else None), source


def _model_info(db: Database, pred: dict | None) -> dict | None:
    if not pred:
        return None
    return {"model_id": pred["model_id"], "mode": pred["mode"], "created_at": pred["created_at"],
            "recorded": P.recorded_info(db, pred)}


def _catalog_row(row: dict) -> dict:
    return {"product_id": row["catalog_product_id"], "position": row["catalog_position"],
            "published": bool(row["catalog_published"]), "is_cover": bool(row["catalog_is_cover"]),
            "note": row["catalog_note"]}


def _image_payload(db: Database, row: dict, pred: dict | None, review: dict | None) -> dict:
    values, confidence, source = _values_and_confidence(pred, review)
    out = image_dict(row)
    out["catalog"] = _catalog_row(row)
    out["values"] = values
    out["confidence"] = confidence
    out["values_source"] = source
    out["model"] = _model_info(db, pred)
    out["prediction"] = P.to_result(pred, db=db) if pred else None
    out["review"] = review
    return out


def _load_images(db: Database, product_id: str) -> list[dict]:
    rows = db.all(
        "SELECT i.*, pi.product_id AS catalog_product_id, pi.position AS catalog_position, "
        "pi.published AS catalog_published, pi.is_cover AS catalog_is_cover, pi.note AS catalog_note "
        "FROM product_images pi JOIN images i ON i.image_id = pi.image_id "
        "WHERE pi.product_id = ? ORDER BY pi.position, pi.created_at, i.rowid", (product_id,))
    ids = [r["image_id"] for r in rows]
    preds, reviews = _latest_predictions(db, ids), _latest_reviews(db, ids)
    images = [_image_payload(db, row, preds.get(row["image_id"]), reviews.get(row["image_id"])) for row in rows]
    # copertina scelta dall'admin per prima, poi la regola delle viste; sort stabile: a parità vale l'inserimento
    images.sort(key=lambda i: (not (i["catalog"]["is_cover"] and i["catalog"]["published"]), _perspective_rank(i["values"])))
    explicit = bool(images) and images[0]["catalog"]["is_cover"] and images[0]["catalog"]["published"]
    for index, image in enumerate(images):
        image["catalog"]["position"] = index   # la posizione mostrata segue sempre la regola
        image["catalog"]["is_cover"] = False
    if explicit:
        images[0]["catalog"]["is_cover"] = True
    else:   # nessuna scelta dell'admin: copertina automatica = la prima foto pubblicata in regola (calcolata a ogni lettura)
        first = next((i for i in images if i["catalog"]["published"]), None)
        if first:
            first["catalog"]["is_cover"] = True
    return images


def _perspective_rank(values: dict[str, str | None]) -> int:
    perspective = values.get("perspective")
    return PERSPECTIVE_ORDER.index(perspective) if perspective in PERSPECTIVE_ORDER else len(PERSPECTIVE_ORDER)


def _summary(db: Database, product: dict) -> dict:
    images = _load_images(db, product["product_id"])
    published = [i for i in images if i["catalog"]["published"]]
    cover = next((i for i in published if i["catalog"]["is_cover"]), None) or (published[0] if published else None)
    if cover is None and images:
        cover = images[0]
    confidences = [i["confidence"] for i in (published or images) if i["confidence"] is not None]
    perspectives = sorted({i["values"].get("perspective") for i in published if i["values"].get("perspective")},
                          key=lambda p: PERSPECTIVE_ORDER.index(p) if p in PERSPECTIVE_ORDER else 99)
    needs_review = sum(1 for i in images
                       if i["values_source"] == "none" or (i["confidence"] is not None and i["confidence"] < 0.6))
    return {
        "product_id": product["product_id"], "title": product["title"], "brand": product["brand"],
        "category": product["category"], "subcategory": product["subcategory"], "product_type": product["product_type"],
        "image_count": len(images), "published_count": len(published),
        "status": "live" if published else ("draft" if images else "empty"),
        "cover": _compact_image(cover) if cover else None,
        "perspectives": perspectives,
        "min_confidence": min(confidences) if confidences else None,
        "needs_review": needs_review,
        "images": images,
    }


def _compact_image(image: dict) -> dict:
    return {
        "image_id": image["image_id"], "file_url": image["file_url"], "thumbnail_url": image["thumbnail_url"],
        "original_filename": image["original_filename"], "width": image["width"], "height": image["height"],
        "values": image["values"], "confidence": image["confidence"],
        "is_cover": image["catalog"]["is_cover"], "position": image["catalog"]["position"],
    }


def _matches_search(product: dict, query: str | None) -> bool:
    if not query:
        return True
    needle = query.strip().lower()
    if not needle:
        return True
    haystack = " ".join(str(product.get(k) or "") for k in
                        ("title", "brand", "category", "subcategory", "product_type")).lower()
    return all(word in haystack for word in needle.split())


def _image_visible(values: dict[str, str | None], confidence: float | None, filters: dict) -> bool:
    if filters["hide_non_photos"] and values.get("subject") in NON_PHOTO_SUBJECTS:
        return False
    if filters["only_white_background"] and values.get("background") != "white":
        return False
    if filters["hide_people"] and values.get("individuals") != "none":
        return False
    if filters["hide_added_text"] and values.get("text_classification") != "none":
        return False
    threshold = float(filters.get("min_confidence") or 0.0)
    if threshold > 0 and (confidence is None or confidence < threshold):
        return False
    return True


def get_filters(db: Database) -> dict:
    row = db.one("SELECT value_json FROM settings WHERE key = ?", (FILTERS_KEY,))
    filters = dict(DEFAULT_FILTERS)
    if row:
        try:
            stored = json.loads(row["value_json"])
            if isinstance(stored, dict):
                for key in DEFAULT_FILTERS:
                    if key in stored:
                        filters[key] = stored[key]
        except (TypeError, ValueError):
            pass
    return filters


def set_filters(db: Database, filters: dict) -> dict:
    clean = {key: filters.get(key, DEFAULT_FILTERS[key]) for key in DEFAULT_FILTERS}
    db.execute("INSERT INTO settings (key, value_json) VALUES (?, ?) "
               "ON CONFLICT(key) DO UPDATE SET value_json = excluded.value_json",
               (FILTERS_KEY, json.dumps(clean)))
    return clean


def _stats(db: Database, summaries: list[dict], filters: dict) -> dict:
    images = [image for summary in summaries for image in summary["images"]]
    published = [i for i in images if i["catalog"]["published"]]
    hidden = [i for i in published if not _image_visible(i["values"], i["confidence"], filters)]
    return {
        "products": len(summaries),
        "live_products": sum(1 for s in summaries if s["status"] == "live"),
        "draft_products": sum(1 for s in summaries if s["status"] == "draft"),
        "empty_products": sum(1 for s in summaries if s["status"] == "empty"),
        "images": len(images),
        "published_images": len(published),
        "hidden_by_rules": len(hidden),
        "visible_images": len(published) - len(hidden),
        "needs_review": sum(s["needs_review"] for s in summaries),
    }


def list_products(db: Database, query: str | None = None, status: str | None = None) -> dict:
    products = db.all("SELECT * FROM products ORDER BY title COLLATE NOCASE, brand COLLATE NOCASE")
    summaries = [_summary(db, product) for product in products]
    summaries = [s for s in summaries if _matches_search(s, query)]
    if status in {"live", "draft", "empty"}:
        summaries = [s for s in summaries if s["status"] == status]
    filters = get_filters(db)
    for summary in summaries:
        summary.pop("images", None)
    return {"products": summaries, "stats": _stats(db, [_summary(db, p) for p in products], filters),
            "filters": filters, "categories": sorted({p["category"] for p in products if p["category"]})}


def product_detail(db: Database, product_id: str) -> dict:
    product = db.one("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if product is None:
        raise ApiError("NOT_FOUND", "Product not found.", {"product_id": product_id})
    summary = _summary(db, product)
    return {"product": {k: v for k, v in summary.items() if k != "images"}, "images": summary["images"],
            "filters": get_filters(db)}


def _next_position(db: Database, product_id: str) -> int:
    row = db.one("SELECT COALESCE(MAX(position), -1) + 1 AS next FROM product_images WHERE product_id = ?",
                 (product_id,))
    return int(row["next"]) if row else 0


def _renumber(db: Database, product_id: str) -> None:
    for index, image in enumerate(_load_images(db, product_id)):   # già nell'ordine della regola
        db.execute("UPDATE product_images SET position = ? WHERE image_id = ?", (index, image["image_id"]))


def _ensure_cover(db: Database, product_id: str) -> None:
    """Salva solo la copertina scelta dall'admin (una sola, e pubblicata); senza scelta decide la regola a ogni lettura."""
    rows = db.all("SELECT image_id, published, is_cover FROM product_images WHERE product_id = ? "
                  "ORDER BY is_cover DESC, position, created_at, rowid", (product_id,))
    cover = next((r for r in rows if r["is_cover"] and r["published"]), None)
    if cover:
        db.execute("UPDATE product_images SET is_cover = 0 WHERE product_id = ? AND image_id <> ?",
                   (product_id, cover["image_id"]))
    else:
        db.execute("UPDATE product_images SET is_cover = 0 WHERE product_id = ?", (product_id,))


def update_image(db: Database, image_id: str, body: dict) -> dict:
    image = db.one("SELECT * FROM images WHERE image_id = ?", (image_id,))
    if image is None:
        raise ApiError("NOT_FOUND", "Image not found.", {"image_id": image_id})
    current = db.one("SELECT * FROM product_images WHERE image_id = ?", (image_id,))
    product_id = body.get("product_id") or (current["product_id"] if current else None)
    if not product_id:
        raise ApiError("VALIDATION_ERROR", "`product_id` is required to add an image to the catalog.")
    if db.one("SELECT 1 FROM products WHERE product_id = ?", (product_id,)) is None:
        raise ApiError("NOT_FOUND", "Product not found.", {"product_id": product_id})
    if current and current["product_id"] != product_id:
        db.execute("DELETE FROM product_images WHERE image_id = ?", (image_id,))
        current = None
    published = body["published"] if "published" in body and body["published"] is not None else (
        bool(current["published"]) if current else False)
    position = body["position"] if "position" in body and body["position"] is not None else (
        current["position"] if current else _next_position(db, product_id))
    is_cover = body["is_cover"] if "is_cover" in body and body["is_cover"] is not None else (
        bool(current["is_cover"]) if current else False)
    note = body["note"] if "note" in body else (current["note"] if current else None)
    if body.get("is_cover") is True:
        published = True
        position = 0
        db.execute("UPDATE product_images SET is_cover = 0 WHERE product_id = ? AND image_id <> ?",
                   (product_id, image_id))
    if not published:
        is_cover = False
    now = now_iso()
    if current is None:
        db.execute(
            "INSERT INTO product_images (image_id, product_id, position, published, is_cover, note, created_at, updated_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (image_id, product_id, position, int(published), int(is_cover), note, now, now))
    else:
        db.execute(
            "UPDATE product_images SET product_id = ?, position = ?, published = ?, is_cover = ?, note = ?, updated_at = ? "
            "WHERE image_id = ?",
            (product_id, position, int(published), int(is_cover), note, now, image_id))
    _renumber(db, product_id)
    _ensure_cover(db, product_id)
    return product_detail(db, product_id)


def create_product(db: Database, body: dict) -> tuple[dict, bool]:
    title = (body.get("title") or "").strip()
    if not title:
        raise ApiError("VALIDATION_ERROR", "`title` is required.")
    product, created = ensure_product(db, title, body.get("brand"), body.get("category"),
                                      body.get("subcategory"), body.get("product_type"))
    return product_detail(db, product["product_id"]), created


def update_product(db: Database, product_id: str, body: dict) -> dict:
    product = db.one("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if product is None:
        raise ApiError("NOT_FOUND", "Product not found.", {"product_id": product_id})
    values = {key: _clean(body.get(key)) if key != "title" else (body.get("title") or "").strip()
              for key in ("title", "brand", "category", "subcategory", "product_type") if key in body}
    if values.get("title") == "":
        raise ApiError("VALIDATION_ERROR", "`title` cannot be empty.")
    if values:
        assignments = ", ".join(f"{key} = ?" for key in values)
        db.execute(f"UPDATE products SET {assignments}, updated_at = ? WHERE product_id = ?",
                   (*values.values(), now_iso(), product_id))
    return product_detail(db, product_id)


def storefront(db: Database, query: str | None = None, category: str | None = None) -> dict:
    filters = get_filters(db)
    products = db.all("SELECT * FROM products ORDER BY title COLLATE NOCASE, brand COLLATE NOCASE")
    cards = []
    total_published = hidden = 0
    categories: dict[str, int] = {}
    for product in products:
        summary = _summary(db, product)
        visible = [i for i in summary["images"]
                   if i["catalog"]["published"] and _image_visible(i["values"], i["confidence"], filters)]
        total_published += summary["published_count"]
        hidden += summary["published_count"] - len(visible)
        if not visible or not _matches_search(summary, query):
            continue
        if summary["category"]:
            categories[summary["category"]] = categories.get(summary["category"], 0) + 1
        if category and (summary["category"] or "") != category:
            continue
        cover = next((i for i in visible if i["catalog"]["is_cover"]), None) or visible[0]
        cards.append({
            "product_id": product["product_id"], "title": product["title"], "brand": product["brand"],
            "category": product["category"], "subcategory": product["subcategory"],
            "product_type": product["product_type"], "image_count": len(visible),
            "cover": _compact_image(cover), "perspectives": summary["perspectives"],
            "min_confidence": min((i["confidence"] for i in visible if i["confidence"] is not None), default=None),
        })
    return {"products": cards, "total": len(cards), "filters": filters,
            "categories": [{"name": name, "count": count} for name, count in sorted(categories.items())],
            "stats": {"published_images": total_published, "hidden_by_rules": hidden,
                      "visible_images": total_published - hidden}}


def storefront_product(db: Database, product_id: str) -> dict:
    product = db.one("SELECT * FROM products WHERE product_id = ?", (product_id,))
    if product is None:
        raise ApiError("NOT_FOUND", "Product not found.", {"product_id": product_id})
    filters = get_filters(db)
    summary = _summary(db, product)
    visible = [i for i in summary["images"]
               if i["catalog"]["published"] and _image_visible(i["values"], i["confidence"], filters)]
    if not visible:
        raise ApiError("NOT_FOUND", "This product has no published image that passes the shop rules.",
                       {"product_id": product_id})
    return {"product": {**{k: v for k, v in summary.items() if k != "images"}, "image_count": len(visible)},
            "images": [_compact_image(i) | {"values": i["values"], "confidence": i["confidence"]} for i in visible],
            "filters": filters}


def seed(db: Database) -> int:
    """Prima esecuzione: crea i prodotti dalle foto già caricate. Una sola foto per prodotto va in vetrina.

    Le altre restano in catalogo come bozze: è il punto di partenza della storia "aggiungi, valuta, pubblica".
    Idempotente: se esiste già un prodotto non fa nulla.
    """
    if db.one("SELECT 1 FROM products LIMIT 1"):
        return 0
    rows = db.all("SELECT * FROM images ORDER BY created_at, rowid")
    seen_sha: set[str] = set()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        if row["sha256"] in seen_sha:
            continue
        seen_sha.add(row["sha256"])
        context = json.loads(row["context_json"] or "{}")
        title = (context.get("product_title") or "").strip()
        if not title:
            continue
        product, _ = ensure_product(db, title, context.get("brand"), context.get("category"),
                                    context.get("subcategory"), context.get("product_type"))
        grouped.setdefault(product["product_id"], []).append(row)
    created = 0
    for product_id, images in grouped.items():
        ids = [row["image_id"] for row in images]
        preds, reviews = _latest_predictions(db, ids), _latest_reviews(db, ids)
        def sort_key(row: dict) -> tuple:
            values, _, source = _values_and_confidence(preds.get(row["image_id"]), reviews.get(row["image_id"]))
            return (0 if source != "none" else 1, _perspective_rank(values), row["created_at"])
        images.sort(key=sort_key)
        for index, row in enumerate(images):
            values, _, source = _values_and_confidence(preds.get(row["image_id"]), reviews.get(row["image_id"]))
            published = index == 0 and source != "none"
            now = now_iso()
            db.execute(
                "INSERT OR IGNORE INTO product_images "
                "(image_id, product_id, position, published, is_cover, note, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?)",
                (row["image_id"], product_id, index, int(published), 0, None, now, now))   # copertina: la decide la regola
            created += 1
        _ensure_cover(db, product_id)
    return created
