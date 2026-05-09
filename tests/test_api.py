import pytest
from fastapi.testclient import TestClient

from quickserve.api.main import api
from quickserve.db.models import init_db

client = TestClient(api)


@pytest.fixture(autouse=True, scope="module")
def setup_db():
    init_db()


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_products_endpoint():
    resp = client.get("/products")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_chat_greeting():
    resp = client.post("/chat", json={"message": "hello"})
    assert resp.status_code == 200
    data = resp.json()
    assert "reply" in data
    assert "intent" in data
    assert "session_id" in data
    assert data["intent"] == "greeting"


def test_chat_faq():
    resp = client.post("/chat", json={"message": "what is your return policy"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["intent"] == "faq"
    assert len(data["reply"]) > 10


def test_chat_session_id_preserved():
    resp = client.post("/chat", json={"message": "hi", "session_id": "test-session-001"})
    assert resp.json()["session_id"] == "test-session-001"


def test_order_not_found():
    resp = client.get("/orders/ORDNOTEXIST")
    assert resp.status_code == 404
