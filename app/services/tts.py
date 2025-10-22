"""
Text-to-Speech (TTS) service using ElevenLabs API.
Optimized for low-latency voice synthesis.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, Union, AsyncGenerator
import io
import base64

import httpx
import aiofiles
from pydub import AudioSegment

from app.config import settings

logger = logging.getLogger(__name__)


class ElevenLabsTTSService:
    """Asynchronous ElevenLabs TTS service optimized for low latency."""
    
    def __init__(self):
        self.api_key = getattr(settings, 'elevenlabs_api_key', None)
        self.base_url = "https://api.elevenlabs.io/v1"
        self.client = None
        self.is_initialized = False
        
        # Default voice settings for optimal performance
        self.default_voice_id = "21m00Tcm4TlvDq8ikWAM"  # Rachel voice
        self.default_model = "eleven_turbo_v2"  # Fastest model
        
        # Performance settings
        self.timeout = 15.0  # 15 second timeout
        self.max_text_length = 500  # Limit text length for speed
        self.chunk_size = 1024 * 8  # 8KB chunks for streaming
        
        # Voice settings for quality/speed balance
        self.voice_settings = {
            "stability": 0.5,
            "similarity_boost": 0.8,
            "style": 0.0,
            "use_speaker_boost": True
        }
    
    async def initialize(self):
        """Initialize ElevenLabs client."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing ElevenLabs TTS service")
            
            if not self.api_key:
                raise ValueError("ElevenLabs API key not configured")
            
            # Initialize HTTP client with optimized settings
            self.client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout),
                limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
                headers={
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": self.api_key
                }
            )
            
            # Test API connection
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("ElevenLabs TTS service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ElevenLabs TTS service: {e}")
            raise
    
    async def _test_connection(self):
        """Test ElevenLabs API connection."""
        try:
            response = await self.client.get(f"{self.base_url}/voices")
            response.raise_for_status()
            logger.info("ElevenLabs API connection test successful")
        except Exception as e:
            logger.error(f"ElevenLabs API connection test failed: {e}")
            raise
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None,
        voice_settings: Optional[Dict[str, Any]] = None,
        output_format: str = "mp3_44100_128"
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model: TTS model to use
            voice_settings: Voice configuration
            output_format: Audio output format
            
        Returns:
            Dictionary with audio data and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Validate and prepare text
            text = await self._prepare_text(text)
            
            # Use defaults if not specified
            voice_id = voice_id or self.default_voice_id
            model = model or self.default_model
            voice_settings = voice_settings or self.voice_settings
            
            # Prepare request payload
            payload = {
                "text": text,
                "model_id": model,
                "voice_settings": voice_settings
            }
            
            # Make API request
            url = f"{self.base_url}/text-to-speech/{voice_id}"
            
            response = await self.client.post(
                url,
                json=payload,
                params={"output_format": output_format}
            )
            
            response.raise_for_status()
            
            # Get audio data
            audio_data = response.content
            processing_time = time.time() - start_time
            
            return {
                "audio_data": audio_data,
                "audio_base64": base64.b64encode(audio_data).decode('utf-8'),
                "format": output_format,
                "voice_id": voice_id,
                "model": model,
                "text_length": len(text),
                "audio_size": len(audio_data),
                "processing_time": processing_time,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Speech synthesis failed: {e}")
            return {
                "audio_data": b"",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming for lower latency.
        
        Args:
            text: Text to convert to speech
            voice_id: ElevenLabs voice ID
            model: TTS model to use
            
        Yields:
            Audio data chunks as they're generated
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            text = await self._prepare_text(text)
            voice_id = voice_id or self.default_voice_id
            model = model or self.default_model
            
            payload = {
                "text": text,
                "model_id": model,
                "voice_settings": self.voice_settings
            }
            
            url = f"{self.base_url}/text-to-speech/{voice_id}/stream"
            
            async with self.client.stream(
                "POST",
                url,
                json=payload,
                params={"output_format": "mp3_44100_128"}
            ) as response:
                response.raise_for_status()
                
                async for chunk in response.aiter_bytes(chunk_size=self.chunk_size):
                    if chunk:
                        yield chunk
                        
        except Exception as e:
            logger.error(f"Streaming synthesis failed: {e}")
            # Yield empty chunk to indicate error
            yield b""
    
    async def _prepare_text(self, text: str) -> str:
        """Prepare text for optimal TTS processing."""
        # Clean and validate text
        text = text.strip()
        
        if not text:
            raise ValueError("Text cannot be empty")
        
        # Limit text length for faster processing
        if len(text) > self.max_text_length:
            text = text[:self.max_text_length]
            # Try to end at a sentence boundary
            last_period = text.rfind('.')
            last_exclamation = text.rfind('!')
            last_question = text.rfind('?')
            
            boundary = max(last_period, last_exclamation, last_question)
            if boundary > self.max_text_length * 0.8:  # If boundary is reasonably close
                text = text[:boundary + 1]
            
            logger.warning(f"Text truncated to {len(text)} characters")
        
        # Clean up text for better pronunciation
        text = self._clean_text_for_speech(text)
        
        return text
    
    def _clean_text_for_speech(self, text: str) -> str:
        """Clean text for better speech synthesis."""
        # Replace common abbreviations and symbols
        replacements = {
            "&": "and",
            "@": "at",
            "#": "number",
            "$": "dollars",
            "%": "percent",
            "w/": "with",
            "w/o": "without",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
            "vs.": "versus",
            "Mr.": "Mister",
            "Mrs.": "Missus",
            "Dr.": "Doctor",
            "Prof.": "Professor"
        }
        
        for abbrev, full in replacements.items():
            text = text.replace(abbrev, full)
        
        # Remove excessive punctuation
        text = text.replace("...", ".")
        text = text.replace("!!", "!")
        text = text.replace("??", "?")
        
        # Ensure proper spacing
        text = " ".join(text.split())
        
        return text
    
    async def get_available_voices(self) -> Dict[str, Any]:
        """Get list of available voices."""
        if not self.is_initialized:
            await self.initialize()
        
        try:
            response = await self.client.get(f"{self.base_url}/voices")
            response.raise_for_status()
            
            voices_data = response.json()
            
            # Format voice information
            voices = []
            for voice in voices_data.get("voices", []):
                voices.append({
                    "voice_id": voice.get("voice_id"),
                    "name": voice.get("name"),
                    "category": voice.get("category"),
                    "description": voice.get("description"),
                    "preview_url": voice.get("preview_url"),
                    "labels": voice.get("labels", {})
                })
            
            return {
                "voices": voices,
                "total": len(voices),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Failed to get voices: {e}")
            return {
                "voices": [],
                "error": str(e),
                "success": False
            }
    
    async def convert_audio_format(
        self,
        audio_data: bytes,
        input_format: str = "mp3",
        output_format: str = "wav"
    ) -> bytes:
        """Convert audio between formats."""
        try:
            loop = asyncio.get_event_loop()
            
            def convert():
                # Load audio with pydub
                audio = AudioSegment.from_file(
                    io.BytesIO(audio_data),
                    format=input_format
                )
                
                # Convert to output format
                output_buffer = io.BytesIO()
                audio.export(output_buffer, format=output_format)
                
                return output_buffer.getvalue()
            
            return await loop.run_in_executor(None, convert)
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            return audio_data  # Return original if conversion fails
    
    async def save_audio_file(
        self,
        audio_data: bytes,
        file_path: str,
        format: str = "mp3"
    ) -> bool:
        """Save audio data to file."""
        try:
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(audio_data)
            
            logger.info(f"Audio saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save audio file: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            
            # Quick synthesis test
            result = await self.synthesize_speech("Hello", output_format="mp3_22050_32")
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "response_time": response_time,
                "default_voice": self.default_voice_id,
                "default_model": self.default_model,
                "initialized": self.is_initialized,
                "test_audio_size": result.get("audio_size", 0)
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized
            }
    
    async def close(self):
        """Close the HTTP client."""
        if self.client:
            await self.client.aclose()


# Global TTS service instance
tts_service = ElevenLabsTTSService()


async def synthesize_speech(
    text: str,
    voice_id: Optional[str] = None,
    model: Optional[str] = None
) -> Dict[str, Any]:
    """Convenience function for speech synthesis."""
    return await tts_service.synthesize_speech(text, voice_id, model)


async def synthesize_streaming(
    text: str,
    voice_id: Optional[str] = None
) -> AsyncGenerator[bytes, None]:
    """Convenience function for streaming synthesis."""
    async for chunk in tts_service.synthesize_streaming(text, voice_id):
        yield chunk


async def initialize_tts() -> bool:
    """Initialize TTS service."""
    try:
        await tts_service.initialize()
        return True
    except Exception as e:
        logger.error(f"TTS initialization failed: {e}")
        return False
