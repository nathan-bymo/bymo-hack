import hashlib
import importlib.util
from pathlib import Path

from demo_api.main import create_app
from imgs import make


def test_image_with_indexed_hash_is_rejected(settings):
    from fastapi.testclient import TestClient

    data = make("JPEG", color=(1, 2, 3))
    guard_file = settings.data_dir / "guard" / "sha256.txt"
    guard_file.parent.mkdir(parents=True)
    guard_file.write_text(hashlib.sha256(data).hexdigest() + "\n")
    with TestClient(create_app(settings)) as client:
        assert client.get("/api/health").json()["guard"] == {"loaded": True, "entries": 1}
        sid = client.post("/api/sessions", json={"name": "g"}).json()["session_id"]
        r = client.post("/api/images", data={"session_id": sid}, files=[("files", ("a.jpg", data, "image/jpeg"))])
        assert r.status_code == 422 and r.json()["error"]["code"] == "FORBIDDEN_IMAGE"
        ok = client.post("/api/images", data={"session_id": sid},
                         files=[("files", ("b.jpg", make("JPEG", color=(9, 9, 9)), "image/jpeg"))])
        assert ok.status_code == 201


def test_build_guard_index_writes_only_hashes(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_guard_index.py"
    spec = importlib.util.spec_from_file_location("bgi", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    kaggle = tmp_path / "k"
    (kaggle / "dev_images").mkdir(parents=True)
    (kaggle / "dev_images" / "x.jpg").write_bytes(b"abc")
    out = tmp_path / "out" / "sha256.txt"
    counts = mod.build(kaggle, out)
    assert counts == {"dev_images": 1, "final_images": 0, "unique_hashes": 1}
    assert out.read_text().strip() == hashlib.sha256(b"abc").hexdigest()
