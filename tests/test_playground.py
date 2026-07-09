import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from playground.server import app, state, fresh_index
from vecstore import HNSWIndex


@pytest.fixture(autouse=True)
def reset_index():
    state["index"] = fresh_index(n=80, seed=7)


client = TestClient(app)


def test_graph_returns_nodes_and_layers():
    data = client.get("/api/graph").json()
    assert len(data["nodes"]) == 80
    assert len(data["layers"]) >= 1
    assert data["entry"] in {n["id"] for n in data["nodes"]}


def test_search_traces_down_to_layer_zero():
    data = client.post(
        "/api/search", json={"x": 50, "y": 50, "ef": 10, "k": 5}
    ).json()
    assert len(data["results"]) == 5
    assert data["trace"][-1]["layer"] == 0
    assert 0 < data["stats"]["visited"] <= data["stats"]["total"]


def test_insert_grows_the_index():
    before = client.get("/api/graph").json()
    added = client.post("/api/insert", json={"x": 10, "y": 90}).json()
    after = client.get("/api/graph").json()
    assert len(after["nodes"]) == len(before["nodes"]) + 1
    assert added["level"] >= 0


def test_index_page_is_served():
    page = client.get("/")
    assert page.status_code == 200
    assert "playground" in page.text


def test_empty_index_does_not_crash():
    state["index"] = HNSWIndex(dim=2, M=8, seed=7)  # no points added
    graph = client.get("/api/graph").json()
    assert graph == {"nodes": [], "layers": [], "entry": None}
    result = client.post("/api/search", json={"x": 50, "y": 50}).json()
    assert result["results"] == []
    assert result["stats"]["total"] == 0
