"""Tests for the customer support LLM agent."""

import pytest

from src.llm.agent import CustomerSupportAgent


class FakeEmbeddingModel:
    def encode(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeCollection:
    def query(self, **kwargs):
        return {
            "documents": [["Returns are allowed within 30 days."]],
            "metadatas": [[{"title": "Return Policy Overview", "category": "returns"}]],
            "distances": [[0.12]],
        }


class FakeOllamaResponse:
    def raise_for_status(self):
        return None

    def json(self):
        return {"response": "You can return eligible items within 30 days."}


class FakeHTTPClient:
    def __init__(self):
        self.payload = None

    async def post(self, path, json):
        self.payload = {"path": path, "json": json}
        return FakeOllamaResponse()

    async def aclose(self):
        return None


@pytest.mark.asyncio
async def test_rag_search_formats_results():
    agent = CustomerSupportAgent({})
    agent.collection = FakeCollection()
    agent.embedding_model = FakeEmbeddingModel()

    result = await agent._rag_search("return policy")

    assert "Return Policy Overview" in result
    assert "Returns are allowed within 30 days" in result


@pytest.mark.asyncio
async def test_process_query_calls_ollama_with_rag_context():
    agent = CustomerSupportAgent({"model": "test-model"})
    agent.collection = FakeCollection()
    agent.embedding_model = FakeEmbeddingModel()
    agent.http_client = FakeHTTPClient()
    agent.is_initialized = True

    result = await agent.process_query("Can I return this?")

    assert result == "You can return eligible items within 30 days."
    assert agent.http_client.payload["path"] == "/api/generate"
    assert "Returns are allowed within 30 days" in agent.http_client.payload["json"]["prompt"]


@pytest.mark.asyncio
async def test_process_query_rejects_empty_text():
    agent = CustomerSupportAgent({})
    agent.is_initialized = True
    agent.http_client = FakeHTTPClient()

    with pytest.raises(ValueError, match="cannot be empty"):
        await agent.process_query(" ")
