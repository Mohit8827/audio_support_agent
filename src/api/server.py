"""
FastAPI Server for Audio Customer Support Agent

This module provides REST API endpoints for testing the audio support pipeline.
Students can use this server to test their implementations via HTTP requests.
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import logging
import os
import time

from src.pipeline import AudioSupportPipeline, create_pipeline


class TextRequest(BaseModel):
    """Request model for text-based queries."""
    text: str = Field(..., min_length=1, max_length=2000)
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    components: Dict[str, bool]
    message: str


class TextResponse(BaseModel):
    """Response model for text queries."""
    response_text: str
    audio_available: bool
    processing_time_ms: int


DEFAULT_CORS_ORIGINS = "http://localhost:8501,http://127.0.0.1:8501"
CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", DEFAULT_CORS_ORIGINS).split(",")
    if origin.strip()
]


app = FastAPI(
    title="Audio Customer Support Agent API",
    description="REST API for testing the STT -> LLM -> TTS pipeline",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline: Optional[AudioSupportPipeline] = None

MAX_AUDIO_UPLOAD_BYTES = int(os.getenv("MAX_AUDIO_UPLOAD_BYTES", str(10 * 1024 * 1024)))
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a", ".webm"}
ALLOWED_AUDIO_CONTENT_TYPES = {
    "audio/wav",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/ogg",
    "audio/flac",
    "audio/x-flac",
    "audio/mp4",
    "audio/webm",
    "application/octet-stream",
}

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """
    Initialize the pipeline on server startup.
    
    Students should configure the pipeline with their API keys and settings.
    """
    global pipeline
    
    try:
        logger.info("Starting Audio Support Agent API server...")
        
        stt_config = {
            "provider": "whisper",
            "model": os.getenv("WHISPER_MODEL", "base"),
            "language": os.getenv("WHISPER_LANGUAGE") or None,
            "device": os.getenv("WHISPER_DEVICE") or None,
            "max_audio_bytes": MAX_AUDIO_UPLOAD_BYTES,
        }

        llm_config = {
            "provider": "ollama",
            "model": os.getenv("OLLAMA_MODEL", "llama3.1:8b"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "temperature": float(os.getenv("OLLAMA_TEMPERATURE", "0.2")),
            "timeout": float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "60")),
            "validate_connection": os.getenv("OLLAMA_VALIDATE_CONNECTION", "true").lower() == "true",
            "max_input_chars": int(os.getenv("MAX_TEXT_INPUT_CHARS", "2000")),
            "rag_n_results": int(os.getenv("RAG_N_RESULTS", "3")),
        }

        tts_config = {
            "provider": "edge-tts",
            "voice": os.getenv("EDGE_TTS_VOICE", "en-US-AriaNeural"),
            "output_format": os.getenv("EDGE_TTS_OUTPUT_FORMAT", "audio-24khz-48kbitrate-mono-mp3"),
            "max_text_chars": int(os.getenv("MAX_TTS_TEXT_CHARS", "4000")),
        }

        pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        logger.info("Pipeline initialized.")
        
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        # Don't raise here to allow server to start for debugging


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup pipeline resources on server shutdown."""
    global pipeline
    
    if pipeline:
        logger.info("Shutting down pipeline...")
        await pipeline.cleanup()
        pipeline = None


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Audio Customer Support Agent API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the status of all pipeline components.
    """
    global pipeline
    
    if not pipeline:
        return HealthResponse(
            status="unhealthy",
            components={
                "pipeline_initialized": False,
                "stt_ready": False,
                "llm_ready": False,
                "tts_ready": False
            },
            message="Pipeline not initialized"
        )
    
    try:
        components = await pipeline.health_check()
        all_healthy = all(components.values())
        
        return HealthResponse(
            status="healthy" if all_healthy else "unhealthy",
            components=components,
            message="All components ready" if all_healthy else "Some components not ready"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return HealthResponse(
            status="error",
            components={},
            message=f"Health check failed: {str(e)}"
        )


@app.post("/chat/text", response_model=TextResponse)
async def chat_text(request: TextRequest):
    """
    Process text query through the LLM agent.
    
    This endpoint allows testing the LLM component without audio processing.
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        start_time = time.perf_counter()
        response_text, response_audio = await pipeline.process_text(
            request.text,
            **(request.parameters or {}),
        )
        processing_time = int((time.perf_counter() - start_time) * 1000)

        return TextResponse(
            response_text=response_text,
            audio_available=bool(response_audio),
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Text processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/audio")
async def chat_audio(audio: UploadFile = File(...)):
    """
    Process audio query through the complete pipeline.
    
    This endpoint handles the full STT -> LLM -> TTS pipeline.
    
    Args:
        audio: Audio file upload (WAV, MP3, etc.)
        
    Returns:
        Audio response as bytes
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        audio_bytes = await _read_and_validate_audio(audio)
        response_audio = await pipeline.process_audio(audio_bytes)

        return Response(
            content=response_audio,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=response.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Audio processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/chat/audio/{text}")
async def text_to_audio(text: str):
    """
    Convert text to audio using TTS.
    
    Useful for testing TTS component independently.
    
    Args:
        text: Text to convert to speech
        
    Returns:
        Audio file as bytes
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        normalized_text = (text or "").strip()
        if not normalized_text:
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        if not pipeline.tts:
            raise HTTPException(status_code=503, detail="TTS not available")

        audio_bytes = await pipeline.tts.synthesize(normalized_text)

        return Response(
            content=audio_bytes,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": f"attachment; filename=tts_output.mp3"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/debug/stt")
async def debug_stt(audio: UploadFile = File(...)):
    """
    Debug endpoint for testing STT component independently.
    
    Args:
        audio: Audio file to transcribe
        
    Returns:
        Transcription result
    """
    global pipeline
    
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        audio_bytes = await _read_and_validate_audio(audio)
        if not pipeline.stt:
            raise HTTPException(status_code=503, detail="STT not available")

        transcription = await pipeline.stt.transcribe(audio_bytes)

        return {"transcription": transcription}
        
    except Exception as e:
        logger.error(f"STT debug failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def _read_and_validate_audio(audio: UploadFile) -> bytes:
    """Read uploads defensively and reject empty, unsupported, or oversized files."""
    filename = audio.filename or ""
    _, extension = os.path.splitext(filename.lower())

    if extension and extension not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported audio file extension")

    if audio.content_type and audio.content_type not in ALLOWED_AUDIO_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported audio content type")

    audio_bytes = await audio.read(MAX_AUDIO_UPLOAD_BYTES + 1)

    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Empty audio file")
    if len(audio_bytes) > MAX_AUDIO_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Audio file is too large")

    return audio_bytes


if __name__ == "__main__":
    import uvicorn
    
    # Development server settings.
    uvicorn.run(
        "src.api.server:app",
        host=os.getenv("SERVER_HOST", "0.0.0.0"),
        port=int(os.getenv("SERVER_PORT", "8000")),
        reload=os.getenv("DEBUG", "true").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
