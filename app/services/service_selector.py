"""
Service selector with intelligent fallback.
Automatically selects best available service (Groq or OpenAI/faster-whisper).
"""

import logging
from typing import Dict, Any, Optional, List, Union
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class ServiceSelector:
    """Intelligent service selector with fallback logic."""
    
    def __init__(self):
        self.stt_service = None
        self.llm_service = None
        self.tts_service = None
        self.is_initialized = False
        
        # Track service health
        self.groq_stt_available = False
        self.groq_llm_available = False
        self.cartesia_tts_available = False
    
    async def initialize(self):
        """Initialize services based on configuration."""
        if self.is_initialized:
            return
        
        logger.info("Initializing service selector...")
        
        # Initialize STT service
        await self._initialize_stt()
        
        # Initialize LLM service
        await self._initialize_llm()
        
        # Initialize TTS service
        await self._initialize_tts()
        
        self.is_initialized = True
        logger.info(f"✅ Service selector initialized (STT: {self.stt_service.__class__.__name__}, LLM: {self.llm_service.__class__.__name__}, TTS: {self.tts_service.__class__.__name__})")
    
    async def _initialize_stt(self):
        """Initialize STT service with fallback."""
        # Try Groq first if enabled
        if settings.use_groq_stt:
            try:
                from app.services.groq_stt import groq_stt_service
                await groq_stt_service.initialize()
                self.stt_service = groq_stt_service
                self.groq_stt_available = True
                logger.info("🚀 Using Groq Whisper for STT (ultra-fast)")
                return
            except Exception as e:
                logger.warning(f"Groq STT initialization failed: {e}. Falling back to faster-whisper.")
        
        # Fallback to faster-whisper
        try:
            from app.services.stt import stt_service
            await stt_service.initialize()
            self.stt_service = stt_service
            logger.info("Using faster-whisper for STT (local)")
        except Exception as e:
            logger.error(f"STT initialization failed: {e}")
            raise
    
    async def _initialize_llm(self):
        """Initialize LLM service with fallback."""
        # Try Groq first if enabled
        if settings.use_groq_llm:
            try:
                from app.services.groq_llm import groq_llm_service
                await groq_llm_service.initialize()
                self.llm_service = groq_llm_service
                self.groq_llm_available = True
                logger.info("🚀 Using Groq Llama 3.1 70B for LLM (ultra-fast)")
                return
            except Exception as e:
                logger.warning(f"Groq LLM initialization failed: {e}. Falling back to OpenAI.")
        
        # Fallback to OpenAI
        try:
            from app.services.llm import llm_service
            await llm_service.initialize()
            self.llm_service = llm_service
            logger.info("Using OpenAI GPT-4o-mini for LLM")
        except Exception as e:
            logger.error(f"LLM initialization failed: {e}")
            raise
    
    async def _initialize_tts(self):
        """Initialize TTS service with fallback."""
        # Try Cartesia first if enabled
        if settings.use_cartesia_tts:
            try:
                from app.services.cartesia_tts import cartesia_tts_service
                await cartesia_tts_service.initialize()
                self.tts_service = cartesia_tts_service
                self.cartesia_tts_available = True
                logger.info("🚀 Using Cartesia Sonic for TTS (ultra-fast)")
                return
            except Exception as e:
                logger.warning(f"Cartesia TTS initialization failed: {e}. Falling back to ElevenLabs.")
        
        # Fallback to ElevenLabs
        try:
            from app.services.tts import tts_service
            await tts_service.initialize()
            self.tts_service = tts_service
            logger.info("Using ElevenLabs for TTS")
        except Exception as e:
            logger.error(f"TTS initialization failed: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, np.ndarray, str],
        language: Optional[str] = "en"
    ) -> Dict[str, Any]:
        """Transcribe audio with automatic fallback."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.stt_service.transcribe_audio(audio_data, language)
        except Exception as e:
            logger.error(f"STT service failed: {e}")
            
            # Try fallback if Groq failed
            if self.groq_stt_available and settings.groq_fallback_to_openai:
                logger.warning("Attempting fallback to faster-whisper...")
                try:
                    from app.services.stt import stt_service
                    await stt_service.initialize()
                    return await stt_service.transcribe_audio(audio_data, language)
                except Exception as fallback_error:
                    logger.error(f"Fallback STT also failed: {fallback_error}")
            
            raise
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """Generate LLM response with automatic fallback."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.llm_service.generate_response(user_message, context, conversation_history)
        except Exception as e:
            logger.error(f"LLM service failed: {e}")
            
            # Try fallback if Groq failed
            if self.groq_llm_available and settings.groq_fallback_to_openai:
                logger.warning("Attempting fallback to OpenAI...")
                try:
                    from app.services.llm import llm_service
                    await llm_service.initialize()
                    return await llm_service.generate_response(user_message, context, conversation_history)
                except Exception as fallback_error:
                    logger.error(f"Fallback LLM also failed: {fallback_error}")
            
            raise
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Synthesize speech with automatic fallback."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            return await self.tts_service.synthesize_speech(text, voice_id)
        except Exception as e:
            logger.error(f"TTS service failed: {e}")
            
            # Try fallback if Cartesia failed
            if self.cartesia_tts_available:
                logger.warning("Attempting fallback to ElevenLabs...")
                try:
                    from app.services.tts import tts_service
                    await tts_service.initialize()
                    return await tts_service.synthesize_speech(text, voice_id)
                except Exception as fallback_error:
                    logger.error(f"Fallback TTS also failed: {fallback_error}")
            
            raise
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None
    ):
        """Synthesize speech with streaming."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            async for chunk in self.tts_service.synthesize_streaming(text, voice_id):
                yield chunk
        except Exception as e:
            logger.error(f"TTS streaming failed: {e}")
            raise
    
    async def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services."""
        return {
            "stt_service": self.stt_service.__class__.__name__ if self.stt_service else "None",
            "llm_service": self.llm_service.__class__.__name__ if self.llm_service else "None",
            "tts_service": self.tts_service.__class__.__name__ if self.tts_service else "None",
            "groq_stt_available": self.groq_stt_available,
            "groq_llm_available": self.groq_llm_available,
            "cartesia_tts_available": self.cartesia_tts_available,
            "fallback_enabled": settings.groq_fallback_to_openai,
            "streaming_enabled": settings.enable_streaming,
            "vad_enabled": settings.enable_vad
        }


# Global service selector instance
service_selector = ServiceSelector()


# Convenience functions
async def transcribe_audio(audio_data: Union[bytes, np.ndarray, str], language: Optional[str] = "en") -> Dict[str, Any]:
    """Transcribe audio using best available service."""
    return await service_selector.transcribe_audio(audio_data, language)


async def generate_response(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Generate response using best available service."""
    return await service_selector.generate_response(user_message, context, conversation_history)


async def synthesize_speech(text: str, voice_id: Optional[str] = None) -> Dict[str, Any]:
    """Synthesize speech using best available service."""
    return await service_selector.synthesize_speech(text, voice_id)


async def synthesize_streaming(text: str, voice_id: Optional[str] = None):
    """Synthesize speech with streaming using best available service."""
    async for chunk in service_selector.synthesize_streaming(text, voice_id):
        yield chunk


async def initialize_services() -> bool:
    """Initialize all services."""
    try:
        await service_selector.initialize()
        return True
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False
