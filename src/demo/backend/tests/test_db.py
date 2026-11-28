from demo_api.db import TABLES, Database


def test_ten_tables_created_in_temp_db(tmp_path):
    db = Database(tmp_path / "x.sqlite")
    db.init()
    assert db.tables() == sorted(TABLES) and len(TABLES) == 10
    db.init()  # idempotente
    db.close()


def test_session_roundtrip(client, settings):
    r = client.post("/api/sessions", json={"name": "Demo 28/11"})
    assert r.status_code == 201
    created = r.json()
    assert created["session_id"].startswith("ses_") and created["image_count"] == 0
    assert client.get(f"/api/sessions/{created['session_id']}").json()["name"] == "Demo 28/11"
    assert [s["session_id"] for s in client.get("/api/sessions").json()["sessions"]] == [created["session_id"]]
    assert (settings.data_dir / "demo.sqlite").exists()


def test_session_not_found_and_validation(client):
    assert client.get("/api/sessions/ses_missing").json()["error"]["code"] == "NOT_FOUND"
    r = client.post("/api/sessions", json={"name": ""})
    assert r.status_code == 422 and r.json()["error"]["code"] == "VALIDATION_ERROR"
