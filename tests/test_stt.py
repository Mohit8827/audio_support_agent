"""
Tests for STT (Speech-to-Text) components.

Students can use these tests to verify their STT implementations.
"""

import pytest
from unittest.mock import Mock
from src.stt.base_stt import BaseSTT, STTService


class TestBaseSTT:
    """Test cases for the base STT interface."""
    
    def test_base_stt_is_abstract(self):
        """Test that BaseSTT cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseSTT()
    
    def test_config_initialization(self):
        """Test that config is properly handled."""
        # Create a concrete implementation for testing
        class TestSTT(BaseSTT):
            async def initialize(self): pass
            async def transcribe(self, audio_bytes: bytes, **kwargs): return ""
            async def cleanup(self): pass
        
        config = {"api_key": "test_key", "model": "test_model"}
        stt = TestSTT(config)
        
        assert stt.config == config
        assert not stt.is_initialized
        assert not stt.is_ready()


class TestSTTService:
    """Test cases for generic STT implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_client = Mock()
        self.config = {
            "client": self.mock_client,
            "model": "base",
            "max_audio_bytes": 1024,
        }
        self.stt = STTService(self.config)
    
    def test_initialization(self):
        """Test STTService initialization."""
        assert self.stt.config == self.config
        assert not self.stt.is_initialized
        assert self.stt.client is None
    
    @pytest.mark.asyncio
    async def test_initialize_without_api_key(self):
        """Test local Whisper configuration does not require an API key."""
        stt = STTService({"client": self.mock_client})
        await stt.initialize()
        assert stt.is_initialized
        assert stt.client is self.mock_client
    
    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful initialization."""
        await self.stt.initialize()
        assert self.stt.is_initialized
        assert self.stt.client is self.mock_client
    
    @pytest.mark.asyncio
    async def test_transcribe_not_initialized(self):
        """Test transcribe fails when not initialized."""
        audio_bytes = b"fake_audio_data"
        
        with pytest.raises(RuntimeError, match="not initialized"):
            await self.stt.transcribe(audio_bytes)
    
    @pytest.mark.asyncio
    async def test_transcribe_success(self):
        """Test successful transcription."""
        audio_bytes = b"fake_audio_data"
        expected_text = "Hello, this is a test transcription"

        self.mock_client.transcribe.return_value = {"text": f" {expected_text} "}
        await self.stt.initialize()
        result = await self.stt.transcribe(audio_bytes)

        assert result == expected_text
    
    @pytest.mark.asyncio
    async def test_transcribe_empty_audio(self):
        """Test transcribe handles empty audio."""
        await self.stt.initialize()
        with pytest.raises(ValueError, match="Audio cannot be empty"):
            await self.stt.transcribe(b"")

    @pytest.mark.asyncio
    async def test_transcribe_oversized_audio(self):
        """Test transcribe rejects oversized uploads."""
        await self.stt.initialize()
        with pytest.raises(ValueError, match="exceeds"):
            await self.stt.transcribe(b"x" * 2048)
    
    @pytest.mark.asyncio
    async def test_cleanup(self):
        """Test cleanup functionality."""
        await self.stt.cleanup()
        assert self.stt.client is None
        assert not self.stt.is_initialized


@pytest.mark.integration
class TestSTTIntegration:
    """Integration tests for STT service (requires actual API key)."""
    
    @pytest.mark.skip(reason="Requires actual STT API key and implementation")
    @pytest.mark.asyncio
    async def test_real_transcription(self):
        """Test with real audio data and API key."""
        # TODO: Students can uncomment and use for integration testing with their chosen service
        # import os
        # 
        # # Example for Deepgram:
        # api_key = os.getenv("DEEPGRAM_API_KEY")  # or your chosen service
        # if not api_key:
        #     pytest.skip("API key not set")
        # 
        # config = {"api_key": api_key, "model": "base"}
        # stt = STTService(config)
        # 
        # try:
        #     await stt.initialize()
        #     
        #     # Load test audio file
        #     with open("test_data/sample_audio.wav", "rb") as f:
        #         audio_bytes = f.read()
        #     
        #     result = await stt.transcribe(audio_bytes)
        #     assert isinstance(result, str)
        #     assert len(result) > 0
        #     
        # finally:
        #     await stt.cleanup()
        pass


# Test utilities for creating mock audio data
def create_mock_audio_bytes(duration_seconds: float = 1.0) -> bytes:
    """Create mock audio data for testing."""
    # Simple mock - in real tests you might want actual audio data
    return b"mock_audio_data" * int(duration_seconds * 100)


def create_test_config(api_key: str = "test_key") -> dict:
    """Create a test configuration."""
    return {
        "client": Mock(),
        "model": "base",
        "max_audio_bytes": 1024,
    }
