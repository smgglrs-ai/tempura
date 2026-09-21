"""Tests for API key authentication middleware."""

import os

import httpx
import pytest

TEMPURA_URL = os.environ.get("TEMPURA_URL", "http://localhost:8080")

pytestmark = pytest.mark.skipif(
    not os.environ.get("TEMPURA_API_KEY"),
    reason="Set TEMPURA_API_KEY to run auth tests",
)


def test_correct_api_key_passes():
    key = os.environ["TEMPURA_API_KEY"]
    r = httpx.get(f"{TEMPURA_URL}/v1/models", headers={"Authorization": f"Bearer {key}"})
    assert r.status_code == 200


def test_wrong_api_key_returns_401():
    r = httpx.get(
        f"{TEMPURA_URL}/v1/models",
        headers={"Authorization": "Bearer wrong-key-xyz"},
    )
    assert r.status_code == 401


def test_no_api_key_returns_401():
    r = httpx.get(f"{TEMPURA_URL}/v1/models")
    assert r.status_code == 401


def test_health_bypasses_auth():
    r = httpx.get(f"{TEMPURA_URL}/health")
    assert r.status_code == 200


def test_ready_bypasses_auth():
    r = httpx.get(f"{TEMPURA_URL}/ready")
    assert r.status_code in (200, 503)


def test_metrics_bypasses_auth():
    r = httpx.get(f"{TEMPURA_URL}/metrics")
    assert r.status_code == 200


def test_x_api_key_header_works():
    key = os.environ["TEMPURA_API_KEY"]
    r = httpx.get(f"{TEMPURA_URL}/v1/models", headers={"X-API-Key": key})
    assert r.status_code == 200
