"""Catalogo e vetrina: pubblicazione, filtri admin e ricerca."""

from __future__ import annotations

import io
import json

from PIL import Image

from demo_api import catalog as C
from demo_api.db import now_iso

DIMS = C.DIMENSIONS


def jpeg(color=(120, 120, 120), size=(80, 60)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format="JPEG")
    return buffer.getvalue()


def new_session(client) -> str:
    return client.post("/api/sessions", json={"name": "Catalogo"}).json()["session_id"]


def upload(client, session_id: str, name: str = "foto.jpg", context: dict | None = None) -> dict:
    data = {"session_id": session_id, "source_kind": "own_photo"}
    if context:
        data["context"] = json.dumps(context)
    tone = sum(name.encode("utf-8"))
    color = (30 + tone % 200, 40 + (tone * 3) % 180, 50 + (tone * 7) % 170)
    response = client.post("/api/images", data=data, files={"files": (name, jpeg(color), "image/jpeg")})
    assert response.status_code == 201, response.text
    return response.json()["images"][0]


def add_prediction(client, image: dict, values: dict, conf: float = 0.9, model_id: str = "apertus-8b-vadapter-lora-heads-final"):
    db = client.app.state.db
    dims = {d: {"label": values.get(d), "confidence": conf, "top_k": [{"label": values.get(d), "p": conf}],
                "confidence_method": "head_softmax"} for d in DIMS}
    db.execute(
        "INSERT INTO predictions (prediction_id, image_id, image_sha256, model_id, model_revision, prompt_version, "
        "use_context, context_hash, mode, recorded_run_id, valid, dimensions_json, raw_output, timing_json, job_id, "
        "created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (f"pred_{image['image_id']}", image["image_id"], image["sha256"], model_id, "test", "v1", 0,
         image["context_hash"], "recorded", None, 1, json.dumps(dims), None, "{}", None, now_iso()))


def create_product(client, title="Ceramic mug", brand="Demo brand", category="Home") -> dict:
    response = client.post("/api/catalog/products", json={"title": title, "brand": brand, "category": category,
                                                          "product_type": "Mug"})
    assert response.status_code == 201, response.text
    return response.json()["product"]


def attach(client, image_id: str, product_id: str, **fields) -> dict:
    response = client.put(f"/api/catalog/images/{image_id}", json={"product_id": product_id, **fields})
    assert response.status_code == 200, response.text
    return response.json()


def test_catalog_publish_and_storefront(client):
    session = new_session(client)
    image = upload(client, session)
    values = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
              "individuals": "none", "text_classification": "none", "text_language": "German"}
    add_prediction(client, image, values)
    product = create_product(client)
    attach(client, image["image_id"], product["product_id"], published=True, is_cover=True)

    catalog = client.get("/api/catalog").json()
    row = next(p for p in catalog["products"] if p["product_id"] == product["product_id"])
    assert row["status"] == "live" and row["published_count"] == 1
    assert row["cover"]["values"]["perspective"] == "front_head_on"
    assert catalog["stats"]["live_products"] == 1

    shop = client.get("/api/storefront").json()
    assert shop["total"] == 1 and shop["products"][0]["title"] == "Ceramic mug"
    detail = client.get(f"/api/storefront/products/{product['product_id']}").json()
    assert detail["images"][0]["values"]["subject"] == "full_product"


def test_filters_change_the_shop(client):
    session = new_session(client)
    image = upload(client, session)
    values = {"subject": "full_product", "background": "white", "perspective": "left_side",
              "individuals": "individual_present", "text_classification": "promotional_text", "text_language": "German"}
    add_prediction(client, image, values)
    product = create_product(client, title="Mug with a model")
    attach(client, image["image_id"], product["product_id"], published=True, is_cover=True)

    assert client.get("/api/storefront").json()["total"] == 1
    client.put("/api/storefront/settings", json={"hide_non_photos": True, "only_white_background": False,
                                                 "hide_people": True, "hide_added_text": False, "min_confidence": 0})
    hidden = client.get("/api/storefront").json()
    assert hidden["total"] == 0 and hidden["stats"]["hidden_by_rules"] == 1
    assert client.get(f"/api/storefront/products/{product['product_id']}").json()["error"]["code"] == "NOT_FOUND"

    client.put("/api/storefront/settings", json={"hide_non_photos": True, "only_white_background": False,
                                                 "hide_people": False, "hide_added_text": True, "min_confidence": 0})
    assert client.get("/api/storefront").json()["total"] == 0

    client.put("/api/storefront/settings", json={"hide_non_photos": True, "only_white_background": True,
                                                 "hide_people": False, "hide_added_text": False, "min_confidence": 0.95})
    assert client.get("/api/storefront").json()["total"] == 0


def test_unpublished_and_cover_rules(client):
    session = new_session(client)
    first = upload(client, session, "one.jpg")
    second = upload(client, session, "two.jpg")
    values = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
              "individuals": "none", "text_classification": "none", "text_language": "none"}
    add_prediction(client, first, values)
    add_prediction(client, second, values | {"perspective": "left_side"})
    product = create_product(client)
    attach(client, first["image_id"], product["product_id"], published=True, is_cover=True)
    attach(client, second["image_id"], product["product_id"], published=True)

    detail = attach(client, second["image_id"], product["product_id"], is_cover=True)
    covers = [i for i in detail["images"] if i["catalog"]["is_cover"]]
    assert [i["image_id"] for i in covers] == [second["image_id"]]
    assert detail["images"][0]["image_id"] == second["image_id"]   # la copertina va per prima

    detail = attach(client, second["image_id"], product["product_id"], published=False)
    assert detail["product"]["published_count"] == 1
    assert [i["image_id"] for i in detail["images"] if i["catalog"]["is_cover"]] == [first["image_id"]]
    assert client.get("/api/storefront").json()["products"][0]["image_count"] == 1


def test_search_and_category(client):
    session = new_session(client)
    image = upload(client, session)
    values = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
              "individuals": "none", "text_classification": "none", "text_language": "none"}
    add_prediction(client, image, values)
    product = create_product(client, title="Espresso cup", brand="Bymo", category="Kitchen")
    attach(client, image["image_id"], product["product_id"], published=True, is_cover=True)

    assert client.get("/api/storefront", params={"q": "espresso"}).json()["total"] == 1
    assert client.get("/api/storefront", params={"q": "bymo kitchen"}).json()["total"] == 1
    assert client.get("/api/storefront", params={"q": "not there"}).json()["total"] == 0
    assert client.get("/api/storefront", params={"category": "Kitchen"}).json()["total"] == 1
    assert client.get("/api/storefront", params={"category": "Garden"}).json()["total"] == 0


def test_seed_publishes_one_photo_per_product_and_keeps_drafts(client):
    session = new_session(client)
    context = {"product_title": "Office chair", "brand": "Demo", "category": "Office"}
    first = upload(client, session, "chair-front.jpg", context)
    second = upload(client, session, "chair-side.jpg", context)
    values = {"subject": "full_product", "background": "white", "perspective": "front_head_on",
              "individuals": "none", "text_classification": "none", "text_language": "none"}
    add_prediction(client, first, values)
    add_prediction(client, second, values | {"perspective": "left_side"})
    db = client.app.state.db
    assert C.seed(db) == 2

    catalog = client.get("/api/catalog").json()
    assert catalog["stats"]["products"] == 1
    product = catalog["products"][0]
    assert product["status"] == "live" and product["published_count"] == 1 and product["image_count"] == 2
    detail = client.get(f"/api/catalog/products/{product['product_id']}").json()
    assert [i["catalog"]["published"] for i in detail["images"]] == [True, False]
    assert detail["images"][0]["values"]["perspective"] == "front_head_on"
    assert client.get("/api/storefront").json()["products"][0]["image_count"] == 1


def test_view_order_rule_applies_everywhere(client):
    """Prima le viste ad angolo a sinistra, poi a destra, poi le altre; senza vista in fondo; la copertina scelta per prima."""
    session = new_session(client)
    values = {"subject": "full_product", "background": "white", "individuals": "none",
              "text_classification": "none", "text_language": "none"}
    order_in = ["top_down", "front_angle_right", "front_head_on", "back_angle_left", "front_angle_left", None]
    product = create_product(client, title="Rule chair", brand="Bymo", category="Furniture")
    ids = {}
    for index, view in enumerate(order_in):
        image = upload(client, session, f"view-{index}.jpg")
        add_prediction(client, image, values | {"perspective": view})
        attach(client, image["image_id"], product["product_id"], published=True)
        ids[view] = image["image_id"]

    def views(images):
        return [i["values"]["perspective"] for i in images]

    expected = ["front_angle_left", "back_angle_left", "front_angle_right", "front_head_on", "top_down", None]
    detail = client.get(f"/api/catalog/products/{product['product_id']}").json()
    assert views(detail["images"]) == expected                      # catalogo
    assert [i["catalog"]["position"] for i in detail["images"]] == list(range(6))
    assert detail["images"][0]["catalog"]["is_cover"]                 # senza scelta dell'admin: la prima vista in regola
    shop = client.get(f"/api/storefront/products/{product['product_id']}").json()
    assert views(shop["images"]) == expected                        # vetrina
    cards = client.get("/api/storefront").json()["products"]
    assert cards[0]["perspectives"] == [v for v in expected if v]   # elenco delle viste nella scheda

    detail = attach(client, ids["top_down"], product["product_id"], is_cover=True)   # copertina scelta dall'admin: va per prima
    assert views(detail["images"]) == ["top_down", "front_angle_left", "back_angle_left", "front_angle_right", "front_head_on", None]
