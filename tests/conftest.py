"""Shared fixtures for FastAPI asynchronous testing."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.banana_mlops.serving.app import app


@pytest.fixture
async def client():
    """Async test client fixture for FastAPI endpoints."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
