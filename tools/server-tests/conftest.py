import os

import httpx
import pytest
from openai import OpenAI

TEMPURA_URL = os.environ.get("TEMPURA_URL", "http://localhost:8080")
LLAMACPP_URL = os.environ.get("LLAMACPP_URL", "")
TEMPURA_API_KEY = os.environ.get("TEMPURA_API_KEY", "")


def pytest_configure(config):
    config.addinivalue_line("markers", "compare: requires LLAMACPP_URL env var")
    config.addinivalue_line("markers", "slow: slow tests, skipped unless --run-slow")


def pytest_collection_modifyitems(config, items):
    if not LLAMACPP_URL:
        skip_compare = pytest.mark.skip(reason="Set LLAMACPP_URL to run compare tests")
        for item in items:
            if "compare" in item.keywords:
                item.add_marker(skip_compare)


@pytest.fixture
def tempura_url():
    return TEMPURA_URL


@pytest.fixture
def openai_client():
    return OpenAI(base_url=f"{TEMPURA_URL}/v1", api_key=TEMPURA_API_KEY or "test")


@pytest.fixture
def http():
    headers = {}
    if TEMPURA_API_KEY:
        headers["Authorization"] = f"Bearer {TEMPURA_API_KEY}"
    return httpx.Client(base_url=TEMPURA_URL, headers=headers, timeout=60)
