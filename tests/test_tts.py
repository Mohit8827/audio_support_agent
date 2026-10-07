"""Tests for Text-to-Speech components."""

import pytest

from src.tts.base_tts import TTSService


class FakeCommunicate:
    def __init__(self, text, voice, output_format):
        self.text = text
        self.voice = voice
        self.output_format = output_format

    async def stream(self):
        yield {"type": "metadata", "data": b""}
        yield {"type": "audio", "data": b"abc"}
        yield {"type": "audio", "data": b"123"}


class FakeEdgeTTS:
    Communicate = FakeCommunicate

    async def list_voices(self):
        return [
            {
                "ShortName": "en-US-AriaNeural",
                "FriendlyName": "Microsoft Aria",
                "Locale": "en-US",
                "Gender": "Female",
            }
        ]


@pytest.mark.asyncio
async def test_synthesize_success():
    tts = TTSService({"client": FakeEdgeTTS(), "max_text_chars": 50})
    await tts.initialize()

    result = await tts.synthesize("Hello")

    assert result == b"abc123"


@pytest.mark.asyncio
async def test_synthesize_rejects_empty_text():
    tts = TTSService({"client": FakeEdgeTTS()})
    await tts.initialize()

    with pytest.raises(ValueError, match="Text cannot be empty"):
        await tts.synthesize(" ")


@pytest.mark.asyncio
async def test_synthesize_rejects_oversized_text():
    tts = TTSService({"client": FakeEdgeTTS(), "max_text_chars": 3})
    await tts.initialize()

    with pytest.raises(ValueError, match="exceeds"):
        await tts.synthesize("too long")


@pytest.mark.asyncio
async def test_get_available_voices():
    tts = TTSService({"client": FakeEdgeTTS()})
    await tts.initialize()

    voices = await tts.get_available_voices()

    assert voices[0]["voice_id"] == "en-US-AriaNeural"
