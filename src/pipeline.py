"""
Audio Customer Support Agent Pipeline

This module orchestrates the complete STT -> LLM -> TTS pipeline.
Students should complete the implementation to connect all components.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

from src.stt.base_stt import BaseSTT, STTService
from src.llm.agent import BaseAgent, CustomerSupportAgent
from src.tts.base_tts import BaseTTS, TTSService


@dataclass
class PipelineConfig:
    """Configuration for the audio support pipeline."""
    stt_config: Dict[str, Any]
    llm_config: Dict[str, Any]
    tts_config: Dict[str, Any]
    enable_logging: bool = True


class AudioSupportPipeline:
    """
    Main pipeline class that orchestrates STT -> LLM -> TTS flow.
    
    This class manages the entire audio processing pipeline for customer support.
    Students should complete the implementation to make it fully functional.
    """
    
    def __init__(self, config: PipelineConfig):
        """
        Initialize the audio support pipeline.
        
        Args:
            config: Pipeline configuration containing settings for all components
        """
        self.config = config
        self.stt: Optional[BaseSTT] = None
        self.llm_agent: Optional[BaseAgent] = None
        self.tts: Optional[BaseTTS] = None
        self.is_initialized = False
        
        if config.enable_logging:
            logging.basicConfig(level=logging.INFO)
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.setLevel(logging.CRITICAL)
    
    async def initialize(self) -> None:
        """
        Initialize all pipeline components.
        
        Steps:
        1. Initialize STT service
        2. Initialize LLM agent
        3. Initialize TTS service
        4. Verify all components are ready
        
        Raises:
            Exception: If any component fails to initialize
        """
        try:
            self.logger.info("Initializing Audio Support Pipeline...")
            
            self.logger.info("Initializing STT service...")
            self.stt = STTService(self.config.stt_config)
            await self.stt.initialize()

            self.logger.info("Initializing LLM agent...")
            self.llm_agent = CustomerSupportAgent(self.config.llm_config)
            await self.llm_agent.initialize()

            self.logger.info("Initializing TTS service...")
            self.tts = TTSService(self.config.tts_config)
            await self.tts.initialize()

            if not all([self.stt.is_ready(), self.llm_agent.is_initialized, self.tts.is_ready()]):
                raise RuntimeError("Some pipeline components failed to initialize")

            self.is_initialized = True
            self.logger.info("Pipeline initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"Pipeline initialization failed: {str(e)}")
            await self.cleanup()
            raise
    
    async def process_audio(self, audio_bytes: bytes, **kwargs) -> bytes:
        """
        Process audio input through the complete pipeline.
        
        This is the main method that handles the STT -> LLM -> TTS flow.
        
        Args:
            audio_bytes: Input audio data
            **kwargs: Additional parameters for processing
            
        Returns:
            bytes: Response audio data
            
        Raises:
            RuntimeError: If pipeline is not initialized
            Exception: If processing fails at any stage
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        if not audio_bytes:
            raise ValueError("Audio input cannot be empty")
        if not self.stt or not self.llm_agent or not self.tts:
            raise RuntimeError("Pipeline components are not available")
        
        try:
            self.logger.info("Converting speech to text...")
            text_input = await self.stt.transcribe(audio_bytes, **kwargs)
            self.logger.info("Speech transcription completed")

            self.logger.info("Processing query with LLM agent...")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            self.logger.info("LLM response generated")

            self.logger.info("Converting response to speech...")
            response_audio = await self.tts.synthesize(agent_response, **kwargs)
            self.logger.info("Audio response generated successfully")
            
            return response_audio
            
        except Exception as e:
            self.logger.error(f"Pipeline processing failed: {str(e)}")
            raise
    
    async def process_text(self, text_input: str, **kwargs) -> Tuple[str, bytes]:
        """
        Process text input (useful for testing without STT).
        
        Args:
            text_input: Text query from user
            **kwargs: Additional parameters
            
        Returns:
            Tuple[str, bytes]: (agent_response_text, response_audio)
        """
        if not self.is_initialized:
            raise RuntimeError("Pipeline not initialized. Call initialize() first.")
        if not text_input or not text_input.strip():
            raise ValueError("Text input cannot be empty")
        if not self.llm_agent or not self.tts:
            raise RuntimeError("Pipeline components are not available")

        try:
            self.logger.info("Processing text query")
            agent_response = await self.llm_agent.process_query(text_input, **kwargs)
            response_audio = await self.tts.synthesize(agent_response, **kwargs)

            return agent_response, response_audio
            
        except Exception as e:
            self.logger.error(f"Text processing failed: {str(e)}")
            raise
    
    async def health_check(self) -> Dict[str, bool]:
        """
        Check the health status of all pipeline components.
        
        Returns:
            Dict[str, bool]: Status of each component
        """
        return {
            "pipeline_initialized": self.is_initialized,
            "stt_ready": self.stt.is_ready() if self.stt else False,
            "llm_ready": self.llm_agent.is_initialized if self.llm_agent else False,
            "tts_ready": self.tts.is_ready() if self.tts else False,
        }
    
    async def cleanup(self) -> None:
        """
        Cleanup all pipeline resources.
        
        This method should be called when the pipeline is no longer needed.
        """
        self.logger.info("Cleaning up pipeline resources...")
        
        try:
            # TODO: Cleanup all components
            if self.stt:
                await self.stt.cleanup()
            if self.llm_agent:
                await self.llm_agent.cleanup()
            if self.tts:
                await self.tts.cleanup()
                
            self.stt = None
            self.llm_agent = None
            self.tts = None
            self.is_initialized = False
            
            self.logger.info("Pipeline cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {str(e)}")
            raise


async def create_pipeline(
    stt_config: Dict[str, Any],
    llm_config: Dict[str, Any],
    tts_config: Dict[str, Any],
    enable_logging: bool = True
) -> AudioSupportPipeline:
    """
    Factory function to create and initialize a pipeline.
    
    Args:
        stt_config: STT configuration
        llm_config: LLM configuration  
        tts_config: TTS configuration
        enable_logging: Whether to enable logging
        
    Returns:
        AudioSupportPipeline: Initialized pipeline instance
    """
    config = PipelineConfig(
        stt_config=stt_config,
        llm_config=llm_config,
        tts_config=tts_config,
        enable_logging=enable_logging
    )
    
    pipeline = AudioSupportPipeline(config)
    await pipeline.initialize()
    
    return pipeline


if __name__ == "__main__":
    """
    Example usage of the pipeline.
    Students can use this for testing their implementation.
    """
    async def main():
        stt_config = {
            "model": "base",
            "max_audio_bytes": 10 * 1024 * 1024,
        }
        
        llm_config = {
            "model": "llama3.1:8b",
            "base_url": "http://localhost:11434",
            "temperature": 0.2,
        }
        
        tts_config = {
            "voice": "en-US-AriaNeural",
        }
        
        # TODO: Create and test pipeline
        # pipeline = await create_pipeline(stt_config, llm_config, tts_config)
        
        # TODO: Test with text input
        # response_text, response_audio = await pipeline.process_text("Hello, I need help with my order")
        # print(f"Response: {response_text}")
        
        # TODO: Cleanup
        # await pipeline.cleanup()
        
        print("Pipeline example configured. Uncomment the sample calls to run it locally.")
    
    asyncio.run(main())
