import json

import pytest

from imgs import animated_gif, animated_webp, make


@pytest.fixture
def session_id(client):
    return client.post("/api/sessions", json={"name": "t"}).json()["session_id"]


def up(client, session_id, *files, **data):
    payload = [("files", f) for f in files]
    return client.post("/api/images", data={"session_id": session_id, **data}, files=payload)


def test_valid_formats_accepted(client, session_id):
    r = up(client, session_id, ("a.jpg", make("JPEG"), "image/jpeg"), ("b.png", make("PNG"), "image/png"),
           ("c.webp", make("WEBP"), "image/webp"))
    assert r.status_code == 201
    body = r.json()
    assert [i["format"] for i in body["images"]] == ["JPEG", "PNG", "WEBP"] and body["rejected"] == []
    assert all(i["image_id"].startswith("img-demo-") for i in body["images"])
    assert client.get(body["images"][0]["thumbnail_url"]).headers["content-type"] == "image/jpeg"


def test_text_named_png_is_415(client, session_id):
    r = up(client, session_id, ("fake.png", b"hello this is text", "image/png"))
    assert r.status_code == 415
    assert r.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert r.json()["error"]["details"]["rejected"][0]["original_filename"] == "fake.png"


def test_declared_mime_must_match_magic_bytes(client, session_id):
    r = up(client, session_id, ("a.png", make("JPEG"), "image/png"))
    assert r.status_code == 415 and r.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"


def test_truncated_jpeg_is_corrupt(client, session_id):
    data = make("JPEG", size=(200, 200))
    r = up(client, session_id, ("t.jpg", data[: len(data) // 2], "image/jpeg"))
    assert r.status_code == 422 and r.json()["error"]["code"] == "CORRUPT_IMAGE"


def test_animated_gif_and_webp_rejected(client, session_id):
    r = up(client, session_id, ("a.gif", animated_gif(), "image/gif"), ("a.webp", animated_webp(), "image/webp"))
    assert r.status_code == 415
    codes = [x["error"]["code"] for x in r.json()["error"]["details"]["rejected"]]
    assert codes == ["UNSUPPORTED_MEDIA_TYPE", "UNSUPPORTED_MEDIA_TYPE"]


def test_file_too_large(client, session_id, settings):
    settings.max_upload_mb = 1
    big = make("PNG", size=(1200, 1200)) + b"\0" * (1024 * 1024)
    r = up(client, session_id, ("big.png", big, "image/png"))
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"


def test_too_many_pixels(client, session_id, settings):
    settings.max_pixels = 100
    r = up(client, session_id, ("a.png", make("PNG", size=(50, 50)), "image/png"))
    assert r.status_code == 413 and r.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_exif_orientation_6_saved_as_png_with_swapped_size(client, session_id):
    r = up(client, session_id, ("o.jpg", make("JPEG", size=(48, 32), exif_orientation=6), "image/jpeg"))
    img = r.json()["images"][0]
    assert img["exif_transposed"] is True and img["format"] == "PNG"
    assert (img["width"], img["height"]) == (32, 48)
    assert client.get(img["file_url"]).headers["content-type"] == "image/png"


def test_same_file_twice_is_duplicate(client, session_id):
    data = make("JPEG")
    first = up(client, session_id, ("a.jpg", data, "image/jpeg")).json()["images"][0]
    second = up(client, session_id, ("b.jpg", data, "image/jpeg")).json()["images"][0]
    assert first["duplicate"] is False and second["duplicate"] is True
    assert first["image_id"] == second["image_id"]
    assert len(client.get("/api/images", params={"session_id": session_id}).json()["images"]) == 1


def test_jpeg_without_exif_stored_byte_identical(client, session_id):
    data = make("JPEG")
    img = up(client, session_id, ("a.jpg", data, "image/jpeg")).json()["images"][0]
    assert img["exif_transposed"] is False
    assert client.get(img["file_url"]).content == data


def test_mixed_batch_keeps_valid_files(client, session_id):
    r = up(client, session_id, ("a.jpg", make("JPEG"), "image/jpeg"), ("x.png", b"nope", "image/png"))
    body = r.json()
    assert len(body["images"]) == 1 and len(body["rejected"]) == 1


def test_context_roundtrip_and_hash(client, session_id):
    ctx = json.dumps({"product_title": "Kettle", "bogus": "x", "brand": " "})
    img = up(client, session_id, ("a.jpg", make("JPEG"), "image/jpeg"), context=ctx).json()["images"][0]
    assert img["context"] == {"product_title": "Kettle"}
    h1 = img["context_hash"]
    r = client.put(f"/api/images/{img['image_id']}/context", json={"brand": "Acme", "product_title": "Kettle"})
    assert r.json()["context"] == {"product_title": "Kettle", "brand": "Acme"} and r.json()["context_hash"] != h1


def test_unknown_session_and_image(client):
    r = client.post("/api/images", data={"session_id": "ses_x"}, files=[("files", ("a.jpg", make(), "image/jpeg"))])
    assert r.status_code == 404 and r.json()["error"]["code"] == "NOT_FOUND"
    assert client.get("/api/images/img-demo-nope").status_code == 404


def test_21_mb_file_is_413_with_default_limit(client, session_id):
    data = make("PNG") + b"\0" * (21 * 1024 * 1024)
    r = up(client, session_id, ("big.png", data, "image/png"))
    assert r.status_code == 413 and r.json()["error"]["code"] == "FILE_TOO_LARGE"
