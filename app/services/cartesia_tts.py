"""
Cartesia.ai TTS service using Sonic model for ultra-low latency.
Achieves 100-200ms latency with WebSocket streaming.
"""

import asyncio
import logging
import time
import base64
from typing import Optional, Dict, Any, AsyncGenerator
import json

import httpx
import websockets

from app.config import settings

logger = logging.getLogger(__name__)


class CartesiaTTSService:
    """Asynchronous Cartesia TTS service optimized for ultra-low latency."""
    
    def __init__(self):
        self.api_key = settings.cartesia_api_key
        self.voice_id = settings.cartesia_voice_id
        self.model = settings.cartesia_model
        self.is_initialized = False
        
        # API endpoints
        self.http_url = "https://api.cartesia.ai/tts/bytes"
        self.ws_url = "wss://api.cartesia.ai/tts/websocket"
        
        # Performance settings
        self.timeout = 10.0
        self.sample_rate = 44100  # Cartesia default
        self.output_format = {
            "container": "raw",
            "encoding": "pcm_f32le",
            "sample_rate": self.sample_rate
        }
    
    async def initialize(self):
        """Initialize Cartesia TTS service."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing Cartesia TTS service")
            
            # Test connection
            await self._test_connection()
            
            self.is_initialized = True
            logger.info(f"✅ Cartesia TTS service initialized successfully (Model: {self.model})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Cartesia TTS service: {e}")
            raise
    
    async def _test_connection(self):
        """Test Cartesia API connection."""
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.http_url,
                    headers={
                        "X-API-Key": self.api_key,
                        "Cartesia-Version": "2024-06-10",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model_id": self.model,
                        "transcript": "Hello",
                        "voice": {"mode": "id", "id": self.voice_id},
                        "output_format": self.output_format
                    }
                )
                
                if response.status_code == 200:
                    logger.info("Cartesia API connection test successful")
                else:
                    raise Exception(f"API test failed: {response.status_code} - {response.text}")
                    
        except Exception as e:
            logger.error(f"Cartesia API connection test failed: {e}")
            raise
    
    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize speech from text using Cartesia (HTTP endpoint).
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Returns:
            Dictionary with audio data and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    self.http_url,
                    headers={
                        "X-API-Key": self.api_key,
                        "Cartesia-Version": "2024-06-10",
                        "Content-Type": "application/json"
                    },
                    json={
                        "model_id": self.model,
                        "transcript": text,
                        "voice": {
                            "mode": "id",
                            "id": voice_id or self.voice_id
                        },
                        "output_format": self.output_format
                    }
                )
                
                if response.status_code != 200:
                    raise Exception(f"TTS failed: {response.status_code} - {response.text}")
                
                audio_data = response.content
                processing_time = time.time() - start_time
                
                logger.info(f"🚀 Cartesia TTS: {processing_time*1000:.0f}ms - '{text[:50]}...'")
                
                return {
                    "audio_data": audio_data,
                    "sample_rate": self.sample_rate,
                    "processing_time": processing_time,
                    "text": text,
                    "success": True,
                    "service": "cartesia"
                }
                
        except Exception as e:
            logger.error(f"Cartesia TTS synthesis failed: {e}")
            return {
                "audio_data": b"",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False,
                "service": "cartesia"
            }
    
    async def synthesize_streaming(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> AsyncGenerator[bytes, None]:
        """
        Synthesize speech with streaming using WebSocket.
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID override
            
        Yields:
            Audio chunks as they're generated
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            async with websockets.connect(
                f"{self.ws_url}?api_key={self.api_key}&cartesia_version=2024-06-10"
            ) as websocket:
                
                # Send synthesis request
                request = {
                    "model_id": self.model,
                    "transcript": text,
                    "voice": {
                        "mode": "id",
                        "id": voice_id or self.voice_id
                    },
                    "output_format": self.output_format,
                    "context_id": f"ctx_{int(time.time() * 1000)}"
                }
                
                await websocket.send(json.dumps(request))
                
                # Receive audio chunks
                first_chunk = True
                start_time = time.time()
                
                async for message in websocket:
                    data = json.loads(message)
                    
                    if data.get("done"):
                        break
                    
                    if "audio" in data:
                        # Decode base64 audio
                        audio_chunk = base64.b64decode(data["audio"])
                        
                        if first_chunk:
                            ttfb = time.time() - start_time
                            logger.info(f"🚀 Cartesia streaming TTFB: {ttfb*1000:.0f}ms")
                            first_chunk = False
                        
                        yield audio_chunk
                    
                    if "error" in data:
                        logger.error(f"Cartesia streaming error: {data['error']}")
                        break
                        
        except Exception as e:
            logger.error(f"Cartesia streaming synthesis failed: {e}")
            yield b""
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            result = await self.synthesize_speech("Hello")
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "model": self.model,
                "voice_id": self.voice_id,
                "response_time": response_time,
                "initialized": self.is_initialized,
                "service": "cartesia"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized,
                "service": "cartesia"
            }


# Global Cartesia TTS service instance
cartesia_tts_service = CartesiaTTSService()


async def test_cartesia_tts():
    """Test Cartesia TTS service."""
    print("\n🧪 Testing Cartesia TTS Service...")
    
    try:
        await cartesia_tts_service.initialize()
        print("✅ Cartesia TTS initialized")
        
        # Test synthesis
        start = time.time()
        result = await cartesia_tts_service.synthesize_speech("Hello, how can I help you today?")
        latency = time.time() - start
        
        print(f"✅ Synthesis completed in {latency*1000:.0f}ms")
        print(f"   Audio size: {len(result.get('audio_data', b''))} bytes")
        
        if latency < 0.2:
            print("🚀 Excellent! Sub-200ms latency achieved!")
        
        # Test streaming
        print("\n🧪 Testing streaming...")
        start = time.time()
        chunks = []
        async for chunk in cartesia_tts_service.synthesize_streaming("This is a streaming test."):
            chunks.append(chunk)
        
        stream_latency = time.time() - start
        print(f"✅ Streaming completed in {stream_latency*1000:.0f}ms")
        print(f"   Chunks received: {len(chunks)}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(test_cartesia_tts())
