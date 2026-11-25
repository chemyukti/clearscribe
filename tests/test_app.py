import io

import numpy as np
import pytest
import soundfile as sf

from app import create_app
from conftest import NEELADRI_FRAMES, FakeModel


def wav_bytes(seconds=1.0):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV")
    buf.seek(0)
    return buf


@pytest.fixture
def client(tmp_path):
    model = FakeModel(NEELADRI_FRAMES)
    app = create_app(model_loader=lambda: model, upload_dir=tmp_path)
    return app.test_client()


def upload(client, glossary="", boost="1.5", name="interview.wav"):
    return client.post("/api/transcribe", content_type="multipart/form-data",
                       data={"audio": (wav_bytes(), name), "glossary": glossary, "boost": boost})


def test_index_page(client):
    page = client.get("/")
    assert page.status_code == 200
    assert b"ClearScribe" in page.data


def test_full_review_flow(client):
    res = upload(client, "Neeladri Road\nAniruddha")
    assert res.status_code == 200
    job = res.get_json()
    assert job["standard"]["text"] == "we met near neela road"
    assert job["assisted"]["text"] == "we met near neeladri road"
    assert [r["status"] for r in job["terms"]] == ["suggested", "not_heard"]
    (s,) = job["suggestions"]
    assert s["suggested_text"] == "Neeladri Road"
    assert job["final_text"] == "we met near neela road"  # unchanged until reviewed

    assert client.get(f"/api/jobs/{job['id']}/audio").status_code == 200

    job = client.post(f"/api/jobs/{job['id']}/suggestions/{s['id']}", json={"decision": "accept"}).get_json()
    assert job["final_text"] == "we met near Neeladri Road"
    export = client.get(f"/api/jobs/{job['id']}/export?format=txt")
    assert export.data.decode().strip() == "we met near Neeladri Road"
    assert "attachment" in export.headers["Content-Disposition"]

    job = client.post(f"/api/jobs/{job['id']}/suggestions/{s['id']}", json={"decision": "undo"}).get_json()
    assert job["final_text"] == "we met near neela road"
    assert "not reviewed" in client.get(f"/api/jobs/{job['id']}/export").data.decode()
    assert client.get(f"/api/jobs/{job['id']}/export?format=json").get_json()["id"] == job["id"]


def test_without_glossary(client):
    job = upload(client).get_json()
    assert job["suggestions"] == [] and job["assisted"] == job["standard"]


def test_validation_errors(client):
    assert upload(client, name="notes.txt").status_code == 400
    too_many = "\n".join(f"Name{chr(65 + i)}" for i in range(21))
    res = upload(client, too_many)
    assert res.status_code == 400 and "at most 20" in res.get_json()["error"]
    assert upload(client, "x", boost="99").status_code == 400
    assert client.get("/api/jobs/nope").status_code == 404

    job = upload(client, "Neeladri Road").get_json()
    bad = client.post(f"/api/jobs/{job['id']}/suggestions/s1", json={"decision": "maybe"})
    assert bad.status_code == 400
