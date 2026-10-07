"""Tests for the end-to-end pipeline orchestration."""

import pytest

from src.pipeline import AudioSupportPipeline, PipelineConfig


class FakeSTT:
    def __init__(self, config):
        self.config = config
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True

    async def transcribe(self, audio_bytes, **kwargs):
        return "What is your return policy?"

    async def cleanup(self):
        self.is_initialized = False

    def is_ready(self):
        return self.is_initialized


class FakeAgent:
    def __init__(self, config):
        self.config = config
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True

    async def process_query(self, text, **kwargs):
        return "You can return eligible items within 30 days."

    async def cleanup(self):
        self.is_initialized = False


class FakeTTS:
    def __init__(self, config):
        self.config = config
        self.is_initialized = False

    async def initialize(self):
        self.is_initialized = True

    async def synthesize(self, text, **kwargs):
        return b"audio"

    async def cleanup(self):
        self.is_initialized = False

    def is_ready(self):
        return self.is_initialized


@pytest.fixture
def patched_pipeline(monkeypatch):
    monkeypatch.setattr("src.pipeline.STTService", FakeSTT)
    monkeypatch.setattr("src.pipeline.CustomerSupportAgent", FakeAgent)
    monkeypatch.setattr("src.pipeline.TTSService", FakeTTS)


@pytest.mark.asyncio
async def test_initialize_and_health_check(patched_pipeline):
    pipeline = AudioSupportPipeline(PipelineConfig({}, {}, {}))

    await pipeline.initialize()
    health = await pipeline.health_check()

    assert health == {
        "pipeline_initialized": True,
        "stt_ready": True,
        "llm_ready": True,
        "tts_ready": True,
    }


@pytest.mark.asyncio
async def test_process_audio_runs_all_stages(patched_pipeline):
    pipeline = AudioSupportPipeline(PipelineConfig({}, {}, {}))
    await pipeline.initialize()

    result = await pipeline.process_audio(b"audio")

    assert result == b"audio"


@pytest.mark.asyncio
async def test_process_text_returns_text_and_audio(patched_pipeline):
    pipeline = AudioSupportPipeline(PipelineConfig({}, {}, {}))
    await pipeline.initialize()

    response_text, response_audio = await pipeline.process_text("Hello")

    assert response_text == "You can return eligible items within 30 days."
    assert response_audio == b"audio"


@pytest.mark.asyncio
async def test_process_text_rejects_empty_text(patched_pipeline):
    pipeline = AudioSupportPipeline(PipelineConfig({}, {}, {}))
    await pipeline.initialize()

    with pytest.raises(ValueError, match="Text input cannot be empty"):
        await pipeline.process_text(" ")
