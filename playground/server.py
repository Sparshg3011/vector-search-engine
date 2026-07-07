"""Tiny demo server: a 2-d index you can poke from the browser and
watch the search hop through the layers. One global index, no auth,
no persistence — it's a playground, not a product."""

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from vecstore import HNSWIndex
from vecstore.trace import traced_search

app = FastAPI(title="vecstore playground")


def fresh_index(n=150, seed=7):
    # M=8 keeps the drawing legible, 2 dims so it can be drawn at all
    index = HNSWIndex(dim=2, M=8, ef_construction=64, seed=seed)
    rng = np.random.default_rng(seed)
    for p in rng.uniform(5.0, 95.0, (n, 2)):
        index.add(p)
    return index


state = {"index": fresh_index()}


class Point(BaseModel):
    x: float
    y: float


class SearchReq(Point):
    ef: int = 12
    k: int = 5


class ResetReq(BaseModel):
    n: int = 150
    seed: int = 7


def graph_json(index):
    top = {}
    for l, layer in enumerate(index._layers):
        for node in layer:
            top[node] = l
    nodes = [
        {"id": int(i), "x": float(v[0]), "y": float(v[1]), "level": top[i]}
        for i, v in enumerate(index.vectors)
    ]
    layers = []
    for layer in index._layers:
        edges = {tuple(sorted((a, b))) for a, links in layer.items() for b in links}
        layers.append(sorted([int(a), int(b)] for a, b in edges))
    return {"nodes": nodes, "layers": layers, "entry": int(index._entry)}


@app.get("/api/graph")
def get_graph():
    return graph_json(state["index"])


@app.post("/api/reset")
def reset(req: ResetReq):
    state["index"] = fresh_index(req.n, req.seed)
    return graph_json(state["index"])


@app.post("/api/insert")
def insert(p: Point):
    index = state["index"]
    node = index.add([p.x, p.y])
    level = sum(1 for layer in index._layers if node in layer) - 1
    return {"id": int(node), "level": int(level)}


@app.post("/api/search")
def search(req: SearchReq):
    index = state["index"]
    ids, dists, trace = traced_search(index, [req.x, req.y], k=req.k, ef=req.ef)
    visited = {v for step in trace for v in step["visited"]}
    return {
        "results": ids,
        "distances": dists,
        "trace": trace,
        "stats": {"visited": len(visited), "total": len(index)},
    }


app.mount(
    "/", StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)
