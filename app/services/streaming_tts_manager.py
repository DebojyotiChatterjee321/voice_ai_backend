"""
Streaming TTS Manager with interrupt support for conversational AI.
Handles progressive audio playback and interruption during bot speech.
"""

import asyncio
import logging
import time
from typing import Optional, AsyncGenerator, Callable, Dict, Any
from enum import Enum
import base64

from app.services.tts import tts_service
from app.services.cartesia_tts import cartesia_tts_service
from app.config import settings

logger = logging.getLogger(__name__)


class PlaybackState(Enum):
    """States for TTS playback."""
    IDLE = "idle"
    STREAMING = "streaming"
    INTERRUPTED = "interrupted"
    COMPLETED = "completed"


class StreamingTTSManager:
    """
    Manager for streaming TTS with interrupt support.
    Handles progressive audio generation and playback control.
    """
    
    def __init__(self, use_cartesia: bool = None):
        """
        Initialize streaming TTS manager.
        
        Args:
            use_cartesia: Whether to use Cartesia TTS (faster). Defaults to settings.
        """
        self.use_cartesia = use_cartesia if use_cartesia is not None else settings.use_cartesia_tts
        self.state = PlaybackState.IDLE
        self.current_task: Optional[asyncio.Task] = None
        self.interrupt_event = asyncio.Event()
        
        # Statistics
        self.total_syntheses = 0
        self.total_interruptions = 0
        self.avg_ttfb = 0.0  # Time to first byte
        
        logger.info(f"StreamingTTSManager initialized (Cartesia: {self.use_cartesia})")
    
    async def synthesize_streaming(
        self,
        text: str,
        callback: Callable[[bytes, bool], None],
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize speech with streaming and send chunks via callback.
        
        Args:
            text: Text to synthesize
            callback: Async function to call with each audio chunk
            voice_id: Optional voice ID override
            
        Returns:
            Dictionary with synthesis results
        """
        if self.state == PlaybackState.STREAMING:
            logger.warning("Already streaming, interrupting previous synthesis")
            await self.interrupt()
        
        self.state = PlaybackState.STREAMING
        self.interrupt_event.clear()
        self.total_syntheses += 1
        
        start_time = time.time()
        ttfb = None
        total_bytes = 0
        chunk_count = 0
        
        try:
            if self.use_cartesia:
                # Use Cartesia streaming WebSocket
                logger.info(f"Starting Cartesia streaming TTS: '{text[:50]}...'")
                
                first_chunk = True
                cartesia_failed = False
                
                async for audio_chunk in cartesia_tts_service.synthesize_streaming(text, voice_id):
                    # Check for interrupt
                    if self.interrupt_event.is_set():
                        logger.info("TTS streaming interrupted")
                        self.state = PlaybackState.INTERRUPTED
                        self.total_interruptions += 1
                        return {
                            "success": False,
                            "interrupted": True,
                            "chunks_sent": chunk_count,
                            "bytes_sent": total_bytes
                        }
                    
                    if first_chunk:
                        ttfb = time.time() - start_time
                        self._update_ttfb(ttfb)
                        logger.info(f"🚀 TTFB: {ttfb*1000:.0f}ms")
                        first_chunk = False
                    
                    # Send chunk via callback
                    is_final = False  # Will be set to True on last chunk
                    await callback(audio_chunk, is_final)
                    
                    total_bytes += len(audio_chunk)
                    chunk_count += 1
                
                # If Cartesia didn't generate any audio, fallback to ElevenLabs
                if chunk_count == 0:
                    logger.warning("Cartesia generated no audio, falling back to ElevenLabs")
                    cartesia_failed = True
                else:
                    # Send final marker
                    await callback(b"", True)
                
                # Fallback to ElevenLabs if Cartesia failed
                if cartesia_failed:
                    logger.info(f"Falling back to ElevenLabs TTS: '{text[:50]}...'")
                    result = await tts_service.synthesize_speech(text, voice_id)
                    
                    if not result["success"]:
                        raise Exception(result.get("error", "TTS synthesis failed"))
                    
                    ttfb = result["processing_time"]
                    self._update_ttfb(ttfb)
                    
                    audio_data = result["audio_data"]
                    total_bytes = len(audio_data)
                    
                    # Send as single chunk
                    await callback(audio_data, True)
                    chunk_count = 1
                
            else:
                # Use ElevenLabs (non-streaming fallback)
                logger.info(f"Starting ElevenLabs TTS: '{text[:50]}...'")
                
                result = await tts_service.synthesize_speech(text, voice_id)
                
                if not result["success"]:
                    raise Exception(result.get("error", "TTS synthesis failed"))
                
                ttfb = result["processing_time"]
                self._update_ttfb(ttfb)
                
                audio_data = result["audio_data"]
                total_bytes = len(audio_data)
                
                # Check for interrupt before sending
                if self.interrupt_event.is_set():
                    logger.info("TTS interrupted before playback")
                    self.state = PlaybackState.INTERRUPTED
                    self.total_interruptions += 1
                    return {
                        "success": False,
                        "interrupted": True,
                        "chunks_sent": 0,
                        "bytes_sent": 0
                    }
                
                # Send as single chunk
                await callback(audio_data, True)
                chunk_count = 1
            
            processing_time = time.time() - start_time
            self.state = PlaybackState.COMPLETED
            
            logger.info(
                f"✅ TTS completed: {processing_time*1000:.0f}ms, "
                f"{chunk_count} chunks, {total_bytes} bytes"
            )
            
            return {
                "success": True,
                "processing_time": processing_time,
                "ttfb": ttfb,
                "chunks_sent": chunk_count,
                "bytes_sent": total_bytes,
                "interrupted": False
            }
            
        except Exception as e:
            logger.error(f"TTS streaming error: {e}")
            self.state = PlaybackState.IDLE
            return {
                "success": False,
                "error": str(e),
                "interrupted": self.interrupt_event.is_set()
            }
        finally:
            self.state = PlaybackState.IDLE
    
    async def synthesize_complete(
        self,
        text: str,
        voice_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Synthesize complete audio (non-streaming).
        
        Args:
            text: Text to synthesize
            voice_id: Optional voice ID
            
        Returns:
            Dictionary with audio data and metadata
        """
        try:
            if self.use_cartesia:
                result = await cartesia_tts_service.synthesize_speech(text, voice_id)
            else:
                result = await tts_service.synthesize_speech(text, voice_id)
            
            if result["success"]:
                self.total_syntheses += 1
            
            return result
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return {
                "success": False,
                "error": str(e),
                "audio_data": b""
            }
    
    async def interrupt(self):
        """Interrupt current TTS streaming."""
        if self.state == PlaybackState.STREAMING:
            logger.info("Interrupting TTS playback")
            self.interrupt_event.set()
            self.state = PlaybackState.INTERRUPTED
            
            # Wait a moment for cleanup
            await asyncio.sleep(0.1)
    
    def is_speaking(self) -> bool:
        """Check if currently generating speech."""
        return self.state == PlaybackState.STREAMING
    
    def _update_ttfb(self, ttfb: float):
        """Update rolling average TTFB."""
        if self.total_syntheses == 1:
            self.avg_ttfb = ttfb
        else:
            # Rolling average
            self.avg_ttfb = (self.avg_ttfb * 0.9) + (ttfb * 0.1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics."""
        return {
            "state": self.state.value,
            "use_cartesia": self.use_cartesia,
            "total_syntheses": self.total_syntheses,
            "total_interruptions": self.total_interruptions,
            "avg_ttfb_ms": self.avg_ttfb * 1000,
            "interruption_rate": (
                self.total_interruptions / self.total_syntheses
                if self.total_syntheses > 0 else 0
            )
        }
    
    def reset(self):
        """Reset the TTS manager."""
        self.state = PlaybackState.IDLE
        self.interrupt_event.clear()
        logger.debug("TTS manager reset")


class TTSChunkEncoder:
    """Utility for encoding TTS audio chunks for WebSocket transmission."""
    
    @staticmethod
    def encode_chunk(audio_data: bytes, sequence: int, is_final: bool) -> Dict[str, Any]:
        """
        Encode audio chunk for WebSocket transmission.
        
        Args:
            audio_data: Raw audio bytes
            sequence: Chunk sequence number
            is_final: Whether this is the final chunk
            
        Returns:
            Dictionary with encoded audio and metadata
        """
        return {
            "type": "audio_chunk",
            "data": base64.b64encode(audio_data).decode("utf-8"),
            "sequence": sequence,
            "is_final": is_final,
            "size": len(audio_data)
        }
    
    @staticmethod
    def decode_chunk(encoded_data: str) -> bytes:
        """
        Decode audio chunk from base64.
        
        Args:
            encoded_data: Base64 encoded audio
            
        Returns:
            Raw audio bytes
        """
        return base64.b64decode(encoded_data)


class TTSQueue:
    """Queue for managing TTS requests in order."""
    
    def __init__(self, max_size: int = 10):
        """
        Initialize TTS queue.
        
        Args:
            max_size: Maximum queue size
        """
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self.current_sequence = 0
    
    async def add(self, text: str, priority: int = 0) -> int:
        """
        Add TTS request to queue.
        
        Args:
            text: Text to synthesize
            priority: Request priority (higher = more important)
            
        Returns:
            Sequence number for this request
        """
        sequence = self.current_sequence
        self.current_sequence += 1
        
        await self.queue.put({
            "sequence": sequence,
            "text": text,
            "priority": priority,
            "timestamp": time.time()
        })
        
        return sequence
    
    async def get(self) -> Optional[Dict[str, Any]]:
        """
        Get next TTS request from queue.
        
        Returns:
            TTS request dictionary or None if queue empty
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            return None
    
    def clear(self):
        """Clear all pending requests."""
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    
    def size(self) -> int:
        """Get current queue size."""
        return self.queue.qsize()


# Global TTS manager instance
streaming_tts_manager = StreamingTTSManager()
