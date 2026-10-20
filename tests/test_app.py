import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app


def test_index_get():
    c = app.test_client()
    r = c.get("/")
    assert r.status_code == 200


def test_predict_3000():
    c = app.test_client()
    r = c.post("/", data={"weight": "3000"})
    assert r.status_code == 200
    assert b"23.37" in r.data or b"23.36" in r.data


def test_empty():
    c = app.test_client()
    r = c.post("/", data={"weight": ""})
    assert b"Masukkan berat" in r.data


def test_invalid():
    c = app.test_client()
    r = c.post("/", data={"weight": "abc"})
    assert b"angka yang valid" in r.data


def test_out_of_range():
    c = app.test_client()
    r = c.post("/", data={"weight": "500"})
    assert b"1.000 dan 6.000" in r.data
    r = c.post("/", data={"weight": "7000"})
    assert b"1.000 dan 6.000" in r.data


def test_boundaries():
    c = app.test_client()
    for w in ("1000", "6000"):
        r = c.post("/", data={"weight": w})
        assert r.status_code == 200
