"""
Base Speech-to-Text (STT) Interface

This module defines the abstract base class for Speech-to-Text implementations.
Students should implement the concrete STT class by inheriting from this base class.

Recommended implementation: Deepgram API (free tier available)
Alternative options: OpenAI Whisper, AssemblyAI, or any other STT service
"""

from abc import ABC, abstractmethod
import asyncio
import os
import tempfile
from typing import Optional, Dict, Any


class BaseSTT(ABC):
    """
    Abstract base class for Speech-to-Text implementations.
    
    This class defines the interface that all STT implementations must follow.
    Students should inherit from this class and implement the abstract methods.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the STT service.
        
        Args:
            config: Configuration dictionary containing API keys, model settings, etc.
                   Example: {"api_key": "your_api_key", "model": "nova-2"}
        """
        self.config = config or {}
        self.is_initialized = False
    
    @abstractmethod
    async def initialize(self) -> None:
        """
        Initialize the STT service (setup API clients, load models, etc.).
        This method should be called before using the STT service.
        
        Raises:
            Exception: If initialization fails
        """
        pass
    
    @abstractmethod
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio bytes to text.
        
        Args:
            audio_bytes: Raw audio data as bytes
            **kwargs: Additional parameters specific to the STT implementation
                     (e.g., language, model, formatting options)
        
        Returns:
            str: The transcribed text
            
        Raises:
            Exception: If transcription fails
        """
        pass
    
    @abstractmethod
    async def cleanup(self) -> None:
        """
        Cleanup resources (close connections, free memory, etc.).
        This method should be called when the STT service is no longer needed.
        """
        pass
    
    def is_ready(self) -> bool:
        """
        Check if the STT service is ready to use.
        
        Returns:
            bool: True if ready, False otherwise
        """
        return self.is_initialized


class STTService(BaseSTT):
    """
    Generic STT implementation template.
    
    Students should complete this implementation using their chosen STT service or pretrained model.
    
    API-based options:
    - Deepgram API (free tier, high accuracy): pip install deepgram-sdk
    - AssemblyAI (API-based): pip install assemblyai
    - Azure Speech Services: pip install azure-cognitiveservices-speech
    - Google Cloud Speech: pip install google-cloud-speech
    
    Pretrained model options (local inference):
    - OpenAI Whisper: pip install openai-whisper (various sizes: tiny, base, small, medium, large)
    - Wav2Vec2 models: pip install transformers torch (Facebook's pretrained models)
    - SpeechRecognition + offline engines: pip install SpeechRecognition pocketsphinx
    - Vosk models: pip install vosk (lightweight, supports many languages)
    - Coqui STT: pip install coqui-stt (open-source, pretrained models available)
    
    Input: audio_bytes (bytes) - Raw audio data
    Output: transcribed_text (str) - The text transcription
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.client = None
        self.model_name = self.config.get("model", "base")
        self.language = self.config.get("language")
        self.max_audio_bytes = int(self.config.get("max_audio_bytes", 10 * 1024 * 1024))
        self.file_suffix = self.config.get("file_suffix", ".wav")
        self.device = self.config.get("device")
        # API-based examples:
        # - For Deepgram: from deepgram import DeepgramClient
        # - For AssemblyAI: import assemblyai
        # Pretrained model examples:
        # - For Whisper: import whisper
        # - For Transformers: from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
        # - For Vosk: import vosk
    
    async def initialize(self) -> None:
        """
        Initialize the Whisper STT service.
        
        Steps:
        1. Get API key from config (if using API service)
        2. Create client instance or load model
        3. Set is_initialized to True
        4. Optionally test the connection
        
        Example for API-based services:
        ```python
        api_key = self.config.get("api_key")
        if not api_key:
            raise ValueError("API key not provided")
        self.client = YourSTTClient(api_key)
        ```
        
        Example for pretrained models (local):
        ```python
        # Whisper
        model_name = self.config.get("model", "base")
        self.client = whisper.load_model(model_name)
        
        # Wav2Vec2 with Transformers
        model_name = "facebook/wav2vec2-base-960h"
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.client = Wav2Vec2ForCTC.from_pretrained(model_name)
        
        # Vosk
        model_path = self.config.get("model_path", "path/to/vosk-model")
        self.client = vosk.Model(model_path)
        ```
        """
        injected_client = self.config.get("client")
        if injected_client is not None:
            self.client = injected_client
            self.is_initialized = True
            return

        try:
            import whisper
        except ImportError as exc:
            raise ImportError(
                "openai-whisper is required for STTService. Install it with "
                "`pip install openai-whisper` and ensure ffmpeg is available."
            ) from exc

        load_kwargs = {}
        if self.device:
            load_kwargs["device"] = self.device

        self.client = await asyncio.to_thread(
            whisper.load_model,
            self.model_name,
            **load_kwargs,
        )
        self.is_initialized = True
    
    async def transcribe(self, audio_bytes: bytes, **kwargs) -> str:
        """
        Transcribe audio with Whisper.
        
        Input: audio_bytes (bytes) - Raw audio data in any supported format
        Output: str - Transcribed text
        
        Steps:
        1. Check if service is initialized
        2. Prepare audio data for your chosen service
        3. Call transcription API/model
        4. Extract and return transcribed text
        5. Handle errors appropriately
        
        Example implementations:
        
        For Deepgram:
        ```python
        response = await self.client.listen.prerecorded.v("1").transcribe_file(
            {"buffer": audio_bytes}, 
            {"model": "nova-2", "smart_format": True}
        )
        return response["results"]["channels"][0]["alternatives"][0]["transcript"]
        ```
        
        For Whisper (pretrained model):
        ```python
        import io
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_file:
            temp_file.write(audio_bytes)
            result = self.client.transcribe(temp_file.name)
            return result["text"]
        ```
        
        For Wav2Vec2 (Transformers):
        ```python
        import torch
        import torchaudio
        # Convert bytes to tensor, resample if needed
        waveform, sample_rate = torchaudio.load(io.BytesIO(audio_bytes))
        inputs = self.processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = self.client(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        return self.processor.decode(predicted_ids[0])
        ```
        
        For AssemblyAI (API):
        ```python
        transcriber = assemblyai.Transcriber()
        transcript = transcriber.transcribe(audio_bytes)
        return transcript.text
        ```
        """
        if not self.is_ready():
            raise RuntimeError("STT service not initialized")

        if not isinstance(audio_bytes, (bytes, bytearray)):
            raise TypeError("audio_bytes must be bytes")
        if not audio_bytes:
            raise ValueError("Audio cannot be empty")
        if len(audio_bytes) > self.max_audio_bytes:
            raise ValueError(f"Audio file exceeds {self.max_audio_bytes} byte limit")

        suffix = kwargs.get("file_suffix", self.file_suffix)
        temp_path = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
                temp_file.write(bytes(audio_bytes))
                temp_path = temp_file.name

            transcribe_kwargs = {
                "fp16": bool(kwargs.get("fp16", False)),
                "verbose": bool(kwargs.get("verbose", False)),
            }
            language = kwargs.get("language", self.language)
            if language:
                transcribe_kwargs["language"] = language

            result = await asyncio.to_thread(
                self.client.transcribe,
                temp_path,
                **transcribe_kwargs,
            )
            transcript = (result.get("text") if isinstance(result, dict) else str(result)).strip()

            if not transcript:
                raise ValueError("No speech could be transcribed from the audio")

            return transcript
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
    
    async def cleanup(self) -> None:
        """
        Release STT resources.
        
        Steps:
        1. Close any open connections
        2. Clear client instance
        3. Set is_initialized to False
        """
        self.client = None
        self.is_initialized = False
