"""
Real-time Voice Activity Detection (VAD) for conversational AI.
Processes audio chunks in real-time to detect speech boundaries.
"""

import asyncio
import logging
import time
from typing import Tuple, Optional, List
from enum import Enum
import numpy as np

from app.services.vad import vad_service
from app.config import settings

logger = logging.getLogger(__name__)


class SpeechState(Enum):
    """States for speech detection state machine."""
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEECH = "speech"
    SPEECH_END = "speech_end"


class RealtimeVADProcessor:
    """
    Real-time VAD processor for continuous audio streams.
    Detects speech boundaries with configurable thresholds.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 200,
        speech_threshold: float = 0.5,
        silence_duration_ms: int = 500,
        min_speech_duration_ms: int = 250
    ):
        """
        Initialize real-time VAD processor.
        
        Args:
            sample_rate: Audio sample rate in Hz
            chunk_duration_ms: Duration of each audio chunk in milliseconds
            speech_threshold: Probability threshold for speech detection (0-1)
            silence_duration_ms: Duration of silence to consider speech ended
            min_speech_duration_ms: Minimum duration to consider valid speech
        """
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.speech_threshold = speech_threshold
        self.silence_duration_ms = silence_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        
        # Calculate samples
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.silence_chunks = int(silence_duration_ms / chunk_duration_ms)
        self.min_speech_chunks = int(min_speech_duration_ms / chunk_duration_ms)
        
        # State management
        self.state = SpeechState.SILENCE
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_counter = 0
        
        # Statistics
        self.total_chunks_processed = 0
        self.speech_segments_detected = 0
        
        logger.info(
            f"RealtimeVAD initialized: chunk={chunk_duration_ms}ms, "
            f"threshold={speech_threshold}, silence={silence_duration_ms}ms"
        )
    
    async def process_chunk(
        self,
        audio_chunk: np.ndarray
    ) -> Tuple[SpeechState, Optional[np.ndarray]]:
        """
        Process a single audio chunk and update speech detection state.
        
        Args:
            audio_chunk: Audio data as numpy array
            
        Returns:
            Tuple of (current_state, speech_segment)
            speech_segment is None unless speech just ended
        """
        self.total_chunks_processed += 1
        
        # Ensure chunk is correct size
        if len(audio_chunk) < self.chunk_samples:
            # Pad with zeros if too short
            audio_chunk = np.pad(
                audio_chunk,
                (0, self.chunk_samples - len(audio_chunk)),
                mode='constant'
            )
        elif len(audio_chunk) > self.chunk_samples:
            # Truncate if too long
            audio_chunk = audio_chunk[:self.chunk_samples]
        
        # Get speech probability using existing VAD service
        is_speech = await self._detect_speech_in_chunk(audio_chunk)
        
        # State machine logic
        if self.state == SpeechState.SILENCE:
            if is_speech:
                self.state = SpeechState.SPEECH_START
                self.speech_buffer = [audio_chunk]
                self.speech_counter = 1
                logger.debug("Speech started")
            return self.state, None
        
        elif self.state == SpeechState.SPEECH_START:
            self.speech_buffer.append(audio_chunk)
            
            if is_speech:
                self.speech_counter += 1
                if self.speech_counter >= self.min_speech_chunks:
                    self.state = SpeechState.SPEECH
                    self.silence_counter = 0
                    logger.debug("Speech confirmed")
            else:
                # False start - reset
                self.state = SpeechState.SILENCE
                self.speech_buffer = []
                self.speech_counter = 0
                logger.debug("False start - reset to silence")
            
            return self.state, None
        
        elif self.state == SpeechState.SPEECH:
            self.speech_buffer.append(audio_chunk)
            
            if is_speech:
                self.silence_counter = 0
            else:
                self.silence_counter += 1
                
                if self.silence_counter >= self.silence_chunks:
                    self.state = SpeechState.SPEECH_END
                    logger.debug("Speech ended")
        
            return self.state, None
        
        elif self.state == SpeechState.SPEECH_END:
            # Return the complete speech segment
            speech_segment = self._get_speech_segment()
            self.speech_segments_detected += 1
            
            logger.info(
                f"Speech segment detected: {len(speech_segment)/self.sample_rate:.2f}s "
                f"(Total segments: {self.speech_segments_detected})"
            )
            
            # Store the state before resetting
            current_state = self.state
            
            # Reset to silence and start fresh
            self.state = SpeechState.SILENCE
            self.speech_buffer = []
            self.speech_counter = 0
            self.silence_counter = 0
            
            return current_state, speech_segment
        
        return self.state, None
    
    async def _detect_speech_in_chunk(self, audio_chunk: np.ndarray) -> bool:
        """
        Detect if audio chunk contains speech.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            True if speech detected, False otherwise
        """
        try:
            # Use existing VAD service
            if not vad_service.is_initialized:
                # Fallback to energy-based detection if VAD not available
                return self._energy_based_detection(audio_chunk)
            
            # Get speech probability
            speech_prob = await vad_service.get_speech_probability(audio_chunk)
            
            return speech_prob >= self.speech_threshold
            
        except Exception as e:
            logger.warning(f"VAD detection failed, using energy fallback: {e}")
            return self._energy_based_detection(audio_chunk)
    
    def _energy_based_detection(self, audio_chunk: np.ndarray) -> bool:
        """
        Simple energy-based speech detection as fallback.
        
        Args:
            audio_chunk: Audio data
            
        Returns:
            True if energy above threshold
        """
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Threshold for speech (adjust based on testing)
        energy_threshold = 0.02
        
        return rms > energy_threshold
    
    def _get_speech_segment(self) -> np.ndarray:
        """
        Get the complete speech segment from buffer.
        
        Returns:
            Concatenated audio of speech segment
        """
        if not self.speech_buffer:
            return np.array([], dtype=np.float32)
        
        # Remove trailing silence chunks
        trim_chunks = max(0, len(self.speech_buffer) - self.silence_chunks)
        speech_data = self.speech_buffer[:trim_chunks] if trim_chunks > 0 else self.speech_buffer
        
        return np.concatenate(speech_data) if speech_data else np.array([], dtype=np.float32)
    
    def reset(self):
        """Reset the VAD processor state."""
        self.state = SpeechState.SILENCE
        self.speech_buffer = []
        self.silence_counter = 0
        self.speech_counter = 0
        logger.debug("VAD processor reset")
    
    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "total_chunks_processed": self.total_chunks_processed,
            "speech_segments_detected": self.speech_segments_detected,
            "current_state": self.state.value,
            "buffer_size": len(self.speech_buffer),
            "config": {
                "sample_rate": self.sample_rate,
                "chunk_duration_ms": self.chunk_duration_ms,
                "speech_threshold": self.speech_threshold,
                "silence_duration_ms": self.silence_duration_ms,
                "min_speech_duration_ms": self.min_speech_duration_ms
            }
        }


class VADChunkBuffer:
    """Buffer for collecting audio chunks before VAD processing."""
    
    def __init__(self, max_duration_seconds: float = 30.0, sample_rate: int = 16000):
        """
        Initialize chunk buffer.
        
        Args:
            max_duration_seconds: Maximum buffer duration
            sample_rate: Audio sample rate
        """
        self.max_samples = int(max_duration_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.buffer: List[np.ndarray] = []
        self.total_samples = 0
    
    def add_chunk(self, chunk: np.ndarray) -> bool:
        """
        Add audio chunk to buffer.
        
        Args:
            chunk: Audio chunk
            
        Returns:
            True if added successfully, False if buffer full
        """
        if self.total_samples + len(chunk) > self.max_samples:
            logger.warning("VAD buffer full, discarding oldest chunks")
            self.clear()
        
        self.buffer.append(chunk)
        self.total_samples += len(chunk)
        return True
    
    def get_audio(self) -> np.ndarray:
        """Get concatenated audio from buffer."""
        if not self.buffer:
            return np.array([], dtype=np.float32)
        return np.concatenate(self.buffer)
    
    def clear(self):
        """Clear the buffer."""
        self.buffer = []
        self.total_samples = 0
    
    def duration_seconds(self) -> float:
        """Get current buffer duration in seconds."""
        return self.total_samples / self.sample_rate


# Factory function for easy initialization
def create_realtime_vad(
    sample_rate: int = 16000,
    chunk_duration_ms: int = 200,
    speech_threshold: float = 0.5,
    silence_duration_ms: int = 500,
    min_speech_duration_ms: int = 250
) -> RealtimeVADProcessor:
    """
    Create a configured real-time VAD processor.
    
    Args:
        sample_rate: Audio sample rate
        chunk_duration_ms: Chunk duration in ms
        speech_threshold: Speech detection threshold
        silence_duration_ms: Silence duration to end speech
        min_speech_duration_ms: Minimum speech duration
        
    Returns:
        Configured RealtimeVADProcessor instance
    """
    return RealtimeVADProcessor(
        sample_rate=sample_rate,
        chunk_duration_ms=chunk_duration_ms,
        speech_threshold=speech_threshold,
        silence_duration_ms=silence_duration_ms,
        min_speech_duration_ms=min_speech_duration_ms
    )
