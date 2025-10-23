"""
Speech-to-Text (STT) service using OpenAI Whisper.
Optimized for low-latency real-time processing.
"""

import asyncio
import logging
import io
import time
from typing import Optional, Dict, Any, Union
import tempfile
import os

from faster_whisper import WhisperModel
import torch
import numpy as np
from pydub import AudioSegment
import aiofiles

from app.config import settings

logger = logging.getLogger(__name__)


class WhisperSTTService:
    """Asynchronous Whisper STT service optimized for low latency."""
    
    def __init__(self):
        self.model = None
        self.model_name = settings.whisper_model
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"  # faster-whisper optimization
        self.is_initialized = False
        
        # Performance optimization settings
        self.max_audio_length = 30  # seconds
        self.sample_rate = 16000
        self.chunk_duration = 5  # seconds for streaming
        
    async def initialize(self):
        """Initialize Whisper model asynchronously."""
        if self.is_initialized:
            return
        
        try:
            logger.info(f"Loading Whisper model '{self.model_name}' on {self.device}")
            start_time = time.time()
            
            # Load faster-whisper model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                lambda: WhisperModel(
                    self.model_name, 
                    device=self.device,
                    compute_type=self.compute_type,
                    cpu_threads=4,  # Optimize CPU usage
                    num_workers=1   # Single worker for lower latency
                )
            )
            
            load_time = time.time() - start_time
            logger.info(f"Whisper model loaded in {load_time:.2f}s")
            
            # Warm up the model with a short audio clip
            await self._warmup_model()
            
            self.is_initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize Whisper model: {e}")
            raise
    
    async def _warmup_model(self):
        """Warm up the model with a short audio clip to reduce first-call latency."""
        try:
            # Generate a short silence for warmup
            warmup_audio = np.zeros(self.sample_rate, dtype=np.float32)
            
            loop = asyncio.get_event_loop()
            # faster-whisper returns segments and info
            await loop.run_in_executor(
                None,
                lambda: list(self.model.transcribe(
                    warmup_audio,
                    language="en",
                    task="transcribe",
                    beam_size=1,  # Faster inference
                    best_of=1
                )[0])  # Get segments
            )
            logger.info("Whisper model warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Model warmup failed: {e}")
    
    async def transcribe_audio(
        self, 
        audio_data: Union[bytes, np.ndarray, str],
        language: Optional[str] = "en",
        task: str = "transcribe"
    ) -> Dict[str, Any]:
        """
        Transcribe audio data to text.
        
        Args:
            audio_data: Audio data (bytes, numpy array, or file path)
            language: Language code (None for auto-detection)
            task: "transcribe" or "translate"
            
        Returns:
            Dictionary with transcription results
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Preprocess audio
            processed_audio = await self._preprocess_audio(audio_data)
            
            # Transcribe in executor to avoid blocking
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self._transcribe_sync,
                processed_audio,
                language,
                task
            )
            
            processing_time = time.time() - start_time
            
            return {
                "text": result["text"].strip(),
                "language": result.get("language", language),
                "segments": result.get("segments", []),
                "processing_time": processing_time,
                "confidence": self._calculate_confidence(result),
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    def _transcribe_sync(self, audio: np.ndarray, language: Optional[str], task: str):
        """Synchronous transcription method using faster-whisper."""
        # faster-whisper returns (segments, info) tuple
        segments, info = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=1,  # Faster inference (use 5 for better quality)
            best_of=1,    # Faster inference
            temperature=0.0,  # Deterministic output
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            no_speech_threshold=0.6,
            condition_on_previous_text=False,  # Faster processing
            vad_filter=True,  # Voice activity detection for speed
            vad_parameters=dict(min_silence_duration_ms=500)
        )
        
        # Convert segments generator to list and build result dict
        segments_list = list(segments)
        return {
            "text": " ".join([segment.text for segment in segments_list]),
            "language": info.language,
            "segments": [
                {
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "avg_logprob": segment.avg_logprob
                }
                for segment in segments_list
            ]
        }
    
    async def _preprocess_audio(self, audio_data: Union[bytes, np.ndarray, str]) -> np.ndarray:
        """Preprocess audio data for optimal Whisper performance."""
        try:
            if isinstance(audio_data, str):
                # File path
                async with aiofiles.open(audio_data, 'rb') as f:
                    audio_bytes = await f.read()
                audio_data = audio_bytes
            
            if isinstance(audio_data, bytes):
                # Convert bytes to numpy array
                audio_data = await self._bytes_to_numpy(audio_data)
            
            # Ensure correct format
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Normalize audio
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Trim silence from beginning and end
            audio_data = self._trim_silence(audio_data)
            
            # Limit audio length for faster processing
            max_samples = self.max_audio_length * self.sample_rate
            if len(audio_data) > max_samples:
                audio_data = audio_data[:max_samples]
                logger.warning(f"Audio truncated to {self.max_audio_length} seconds")
            
            return audio_data
            
        except Exception as e:
            logger.error(f"Audio preprocessing failed: {e}")
            raise
    
    async def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """Convert audio bytes to numpy array."""
        try:
            # Use pydub to handle various audio formats
            loop = asyncio.get_event_loop()
            
            def convert_audio():
                audio_segment = AudioSegment.from_file(io.BytesIO(audio_bytes))
                
                # Convert to mono and resample
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_frame_rate(self.sample_rate)
                
                # Convert to numpy array
                samples = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)
                
                # Normalize to [-1, 1]
                if audio_segment.sample_width == 2:  # 16-bit
                    samples = samples / 32768.0
                elif audio_segment.sample_width == 4:  # 32-bit
                    samples = samples / 2147483648.0
                
                return samples
            
            return await loop.run_in_executor(None, convert_audio)
            
        except Exception as e:
            logger.error(f"Audio conversion failed: {e}")
            raise
    
    def _trim_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """Trim silence from beginning and end of audio."""
        # Find non-silent parts
        non_silent = np.abs(audio) > threshold
        
        if not np.any(non_silent):
            return audio  # All silence, return as-is
        
        # Find first and last non-silent samples
        first_sound = np.argmax(non_silent)
        last_sound = len(audio) - np.argmax(non_silent[::-1]) - 1
        
        return audio[first_sound:last_sound + 1]
    
    def _calculate_confidence(self, result: Dict) -> float:
        """Calculate average confidence from Whisper segments."""
        segments = result.get("segments", [])
        if not segments:
            return 0.0
        
        # Whisper doesn't provide confidence directly, estimate from logprobs
        total_logprob = sum(segment.get("avg_logprob", -1.0) for segment in segments)
        avg_logprob = total_logprob / len(segments)
        
        # Convert logprob to confidence (rough approximation)
        confidence = max(0.0, min(1.0, (avg_logprob + 1.0) / 1.0))
        return confidence
    
    async def transcribe_stream(self, audio_chunks: list) -> Dict[str, Any]:
        """
        Transcribe streaming audio chunks.
        
        Args:
            audio_chunks: List of audio chunk bytes
            
        Returns:
            Transcription result
        """
        try:
            # Combine chunks
            combined_audio = b''.join(audio_chunks)
            
            # Transcribe combined audio
            result = await self.transcribe_audio(combined_audio)
            
            return result
            
        except Exception as e:
            logger.error(f"Stream transcription failed: {e}")
            return {
                "text": "",
                "error": str(e),
                "success": False
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Quick transcription test
            test_audio = np.zeros(self.sample_rate, dtype=np.float32)  # 1 second of silence
            
            start_time = time.time()
            result = await self.transcribe_audio(test_audio)
            response_time = time.time() - start_time
            
            return {
                "status": "healthy" if result["success"] else "unhealthy",
                "model": self.model_name,
                "device": self.device,
                "response_time": response_time,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized
            }


# Global STT service instance
stt_service = WhisperSTTService()


async def transcribe_audio(
    audio_data: Union[bytes, np.ndarray, str],
    language: Optional[str] = "en"
) -> Dict[str, Any]:
    """Convenience function for audio transcription."""
    return await stt_service.transcribe_audio(audio_data, language)


async def initialize_stt() -> bool:
    """Initialize STT service."""
    try:
        await stt_service.initialize()
        return True
    except Exception as e:
        logger.error(f"STT initialization failed: {e}")
        return False
