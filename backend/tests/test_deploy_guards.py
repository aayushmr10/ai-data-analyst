"""Protections for a public deployment: each test proves a guard actually blocks."""
import io
import time

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app import guards
from app.llm import config
from app.main import app
from app.static_site import mount_frontend
from app.store import DatasetStore, store

from tests.test_agent import FakeClient, text  # scripted model

client = TestClient(app)


def csv_bytes(n_rows=50):
    return ("a,b\n" + "\n".join(f"{i},{i * 2}" for i in range(n_rows))).encode()


@pytest.fixture()
def dataset(sales_df):
    s = store.create("x.csv", sales_df)
    yield s
    store.delete(s.id)


@pytest.fixture()
def ai_on():
    """Model that answers instantly; lets us test the guards without tool loops."""
    def make():
        return FakeClient([lambda kw: [text("ok")]] * 50)
    config.set_client_override(make())
    yield
    config.set_client_override(None)


def chat(ds, **headers):
    return client.post(f"/api/datasets/{ds.id}/chat", json={"message": "hi"}, headers=headers)


# ------------------------------------------------------------------ upload cap
def test_upload_over_limit_is_rejected_with_413(monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    big = b"a,b\n" + b"1,2\n" * 400_000                      # ~1.6 MB
    r = client.post("/api/datasets/upload", files={"file": ("big.csv", io.BytesIO(big))})
    assert r.status_code == 413 and "1 MB limit" in r.json()["detail"]


def test_body_streaming_cap_applies_even_without_content_length(monkeypatch):
    """The handler itself must refuse >limit bytes (middleware only sees declared sizes)."""
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    monkeypatch.setattr(guards, "max_upload_bytes", lambda: 1_000)   # handler limit tiny, middleware slack ok
    r = client.post("/api/datasets/upload", files={"file": ("m.csv", io.BytesIO(csv_bytes(2000)))})
    assert r.status_code == 413


def test_normal_upload_still_works():
    r = client.post("/api/datasets/upload", files={"file": ("ok.csv", io.BytesIO(csv_bytes()))})
    assert r.status_code == 200
    store.delete(r.json()["dataset_id"])


# ------------------------------------------------------------------ access code
def test_ai_open_when_no_code_configured(dataset, ai_on, monkeypatch):
    monkeypatch.delenv("APP_ACCESS_CODE", raising=False)
    assert client.get("/api/llm/status").json()["requires_code"] is False
    assert chat(dataset).status_code == 200


def test_access_code_blocks_missing_and_wrong_codes(dataset, ai_on, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "s3cret")
    assert client.get("/api/llm/status").json()["requires_code"] is True
    assert chat(dataset).status_code == 403
    assert chat(dataset, **{"X-Access-Code": "nope"}).status_code == 403
    assert chat(dataset, **{"X-Access-Code": "s3cret"}).status_code == 200


def test_access_code_covers_every_costly_endpoint(dataset, ai_on, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "s3cret")
    for path, body in [("chat", {"message": "hi"}), ("briefing", {}), ("root-cause", {"question": "why drop?"})]:
        r = client.post(f"/api/datasets/{dataset.id}/{path}", json=body)
        assert r.status_code == 403, path


def test_free_endpoints_stay_open_with_code_set(dataset, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "s3cret")
    assert client.get(f"/api/datasets/{dataset.id}/chat/suggestions").status_code == 200
    assert client.get("/api/health").status_code == 200


def test_brute_forcing_the_code_gets_locked_out(dataset, ai_on, monkeypatch):
    monkeypatch.setenv("APP_ACCESS_CODE", "s3cret")
    codes = [chat(dataset, **{"X-Access-Code": f"guess{i}"}).status_code for i in range(12)]
    assert codes[:10] == [403] * 10 and codes[10:] == [429, 429]
    assert chat(dataset, **{"X-Access-Code": "s3cret"}).status_code == 200   # real users aren't punished by the lockout


# ------------------------------------------------------------------ rate limits
def test_per_ip_hourly_limit(dataset, ai_on, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "3")
    assert [chat(dataset).status_code for _ in range(5)] == [200, 200, 200, 429, 429]
    assert "3 per hour" in chat(dataset).json()["detail"]


def test_global_daily_cap_holds_even_across_different_ips(dataset, ai_on, monkeypatch):
    """The protection that doesn't depend on trusting X-Forwarded-For."""
    monkeypatch.setenv("AI_DAILY_CAP", "4")
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "100")
    codes = [chat(dataset, **{"X-Forwarded-For": f"10.0.0.{i}"}).status_code for i in range(6)]
    assert codes == [200, 200, 200, 200, 429, 429]


def test_limits_can_be_disabled_with_zero(dataset, ai_on, monkeypatch):
    monkeypatch.setenv("AI_RATE_LIMIT_PER_HOUR", "0")
    monkeypatch.setenv("AI_DAILY_CAP", "0")
    assert all(chat(dataset).status_code == 200 for _ in range(40))


def test_window_expires():
    lim = guards._Limiter()
    now = time.time()
    import os
    os.environ["AI_RATE_LIMIT_PER_HOUR"] = "1"
    try:
        lim.check("1.2.3.4", now)
        with pytest.raises(Exception):
            lim.check("1.2.3.4", now + 10)
        lim.check("1.2.3.4", now + 3601)               # an hour later: allowed again
    finally:
        del os.environ["AI_RATE_LIMIT_PER_HOUR"]


# ------------------------------------------------------------------ session store bounds
def test_sessions_expire_after_ttl(sales_df, monkeypatch):
    monkeypatch.setenv("SESSION_TTL_MIN", "1")
    st = DatasetStore()
    s = st.create("a.csv", sales_df.head(5))
    assert st.get(s.id) is s
    s.last_access -= 61                                  # idle for 61 seconds
    assert st.get(s.id) is None


def test_activity_keeps_a_session_alive(sales_df, monkeypatch):
    monkeypatch.setenv("SESSION_TTL_MIN", "1")
    st = DatasetStore()
    s = st.create("a.csv", sales_df.head(5))
    s.last_access -= 50
    assert st.get(s.id) is s                             # touch resets the idle clock
    s.last_access -= 50
    assert st.get(s.id) is s


def test_max_sessions_evicts_least_recently_used(sales_df, monkeypatch):
    monkeypatch.setenv("MAX_SESSIONS", "3")
    st = DatasetStore()
    a = st.create("a", sales_df.head(5)); b = st.create("b", sales_df.head(5)); c = st.create("c", sales_df.head(5))
    a.last_access -= 300; b.last_access -= 100           # a is stalest
    st.get(c.id)
    d = st.create("d", sales_df.head(5))
    assert st.get(a.id) is None and st.get(b.id) and st.get(c.id) and st.get(d.id)
    assert len(st.list_ids()) == 3


def test_evicted_session_returns_actionable_404(sales_df):
    s = store.create("gone.csv", sales_df.head(5))
    store.delete(s.id)
    r = client.get(f"/api/datasets/{s.id}/profile")
    assert r.status_code == 404 and r.json()["detail"].startswith("Dataset not found")


# ------------------------------------------------------------------ single-container static serving
@pytest.fixture()
def site(tmp_path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "index.html").write_text("<html>SPA-INDEX</html>")
    (tmp_path / "assets" / "app.js").write_text("console.log('hi')")
    (tmp_path / "favicon.svg").write_text("<svg/>")
    (tmp_path.parent / "secret.txt").write_text("TOP-SECRET")
    a = FastAPI()

    @a.get("/api/health")
    def health():
        return {"status": "ok"}

    assert mount_frontend(a, str(tmp_path)) is True
    return TestClient(a)


def test_static_serves_index_assets_and_spa_fallback(site):
    assert "SPA-INDEX" in site.get("/").text
    assert "SPA-INDEX" in site.get("/some/deep/link").text            # client-side routing fallback
    assert site.get("/assets/app.js").text == "console.log('hi')"
    assert site.get("/favicon.svg").text == "<svg/>"


def test_static_never_shadows_the_api(site):
    assert site.get("/api/health").json() == {"status": "ok"}
    r = site.get("/api/does-not-exist")
    assert r.status_code == 404 and "SPA-INDEX" not in r.text          # JSON 404, not the HTML shell


@pytest.mark.parametrize("path", ["/../secret.txt", "/%2e%2e/secret.txt", "/assets/../../secret.txt", "/..%2fsecret.txt"])
def test_static_blocks_path_traversal(site, path):
    r = site.get(path)
    assert "TOP-SECRET" not in r.text


def test_mount_is_a_noop_without_a_built_frontend(tmp_path):
    assert mount_frontend(FastAPI(), str(tmp_path)) is False
