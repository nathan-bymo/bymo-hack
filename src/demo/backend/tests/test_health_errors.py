def test_health_ok(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["gpu_local"] is False
    assert r.headers["X-Request-Id"]


def test_unknown_api_path_uses_error_schema(client):
    r = client.get("/api/does-not-exist")
    assert r.status_code == 404
    error = r.json()["error"]
    assert error["code"] == "NOT_FOUND" and error["request_id"] == r.headers["X-Request-Id"]
    assert set(error) == {"code", "message", "details", "request_id"}


def test_request_id_is_echoed(client):
    r = client.get("/api/health", headers={"X-Request-Id": "abc"})
    assert r.headers["X-Request-Id"] == "abc"


def test_logs_do_not_contain_key(client, settings, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "secret-value-for-test")
    client.get("/api/health")
    log = settings.data_dir / "logs" / "api.log"
    assert "secret-value-for-test" not in (log.read_text() if log.exists() else "")
