"""
Groq STT service using Whisper-large-v3 for ultra-fast transcription.
Cloud-based processing with ~100-200ms latency.
"""

import asyncio
import logging
import time
import tempfile
import os
from typing import Optional, Dict, Any, Union

from groq import AsyncGroq
import httpx
import numpy as np
from pydub import AudioSegment
import io

from app.config import settings

logger = logging.getLogger(__name__)


class GroqSTTService:
    """Asynchronous Groq STT service optimized for ultra-low latency."""
    
    def __init__(self):
        self.client = None
        self.model = settings.groq_stt_model
        self.is_initialized = False
        
        # Performance settings
        self.timeout = 10.0
        self.sample_rate = 16000
    
    async def initialize(self):
        """Initialize Groq client."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing Groq STT service")
            
            # Initialize async client
            self.client = AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=httpx.Timeout(self.timeout),
                max_retries=2
            )
            
            self.is_initialized = True
            logger.info(f"✅ Groq STT service initialized successfully (Model: {self.model})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq STT service: {e}")
            raise
    
    async def transcribe_audio(
        self,
        audio_data: Union[bytes, np.ndarray, str],
        language: Optional[str] = "en"
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text using Groq Whisper.
        
        Args:
            audio_data: Audio data (bytes, numpy array, or file path)
            language: Language code
            
        Returns:
            Dictionary with transcription results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Convert audio to file format Groq expects
            audio_file_path = await self._prepare_audio_file(audio_data)
            
            # Transcribe using Groq
            with open(audio_file_path, "rb") as audio_file:
                result = await self.client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model,
                    language=language,
                    response_format="json",
                    temperature=0.0
                )
            
            processing_time = time.time() - start_time
            
            logger.info(f"🚀 Groq STT: {processing_time*1000:.0f}ms - '{result.text[:50]}...'")
            
            # Clean up temp file
            try:
                os.unlink(audio_file_path)
            except:
                pass
            
            return {
                "text": result.text.strip(),
                "language": language,
                "processing_time": processing_time,
                "confidence": 1.0,  # Groq doesn't provide confidence
                "success": True,
                "service": "groq"
            }
            
        except Exception as e:
            logger.error(f"Groq transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False,
                "service": "groq"
            }
    
    async def _prepare_audio_file(self, audio_data: Union[bytes, np.ndarray, str]) -> str:
        """Prepare audio file for Groq API."""
        try:
            # If already a file path, return it
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                return audio_data
            
            # Convert numpy array to bytes
            if isinstance(audio_data, np.ndarray):
                audio_data = await self._numpy_to_bytes(audio_data)
            
            # Create temporary WAV file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_path = temp_file.name
            
            # Convert to WAV format using pydub
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._convert_to_wav,
                audio_data,
                temp_path
            )
            
            return temp_path
            
        except Exception as e:
            logger.error(f"Audio file preparation failed: {e}")
            raise
    
    def _convert_to_wav(self, audio_bytes: bytes, output_path: str):
        """Convert audio bytes to WAV file."""
        try:
            # Load audio with pydub
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
            
            # Convert to mono and resample if needed
            audio_segment = audio_segment.set_channels(1)
            audio_segment = audio_segment.set_frame_rate(self.sample_rate)
            
            # Export as WAV
            audio_segment.export(output_path, format="wav")
            
        except Exception as e:
            logger.error(f"WAV conversion failed: {e}")
            raise
    
    async def _numpy_to_bytes(self, audio_array: np.ndarray) -> bytes:
        """Convert numpy array to audio bytes."""
        try:
            # Normalize to int16
            if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
                audio_array = (audio_array * 32767).astype(np.int16)
            elif audio_array.dtype != np.int16:
                audio_array = audio_array.astype(np.int16)
            
            # Create WAV file in memory
            loop = asyncio.get_event_loop()
            
            def create_wav():
                from scipy.io import wavfile
                buffer = io.BytesIO()
                wavfile.write(buffer, self.sample_rate, audio_array)
                return buffer.getvalue()
            
            return await loop.run_in_executor(None, create_wav)
            
        except Exception as e:
            logger.error(f"Numpy to bytes conversion failed: {e}")
            raise
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Generate test audio
            test_audio = np.zeros(self.sample_rate, dtype=np.float32)
            
            start_time = time.time()
            result = await self.transcribe_audio(test_audio)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "model": self.model,
                "response_time": response_time,
                "initialized": self.is_initialized,
                "service": "groq"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized,
                "service": "groq"
            }


# Global Groq STT service instance
groq_stt_service = GroqSTTService()


async def test_groq_stt():
    """Test Groq STT service."""
    print("\n🧪 Testing Groq STT Service...")
    
    try:
        await groq_stt_service.initialize()
        print("✅ Groq STT initialized")
        
        # Generate test audio (1 second of silence)
        test_audio = np.zeros(16000, dtype=np.float32)
        
        start = time.time()
        result = await groq_stt_service.transcribe_audio(test_audio)
        latency = time.time() - start
        
        print(f"✅ Transcription completed in {latency*1000:.0f}ms")
        print(f"   Text: '{result.get('text', '')}'")
        
        if latency < 0.2:
            print("🚀 Excellent! Sub-200ms latency achieved!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_groq_stt())
