"""
Voice Activity Detection (VAD) using Silero VAD.
Detects speech segments and removes silence for faster processing.
"""

import asyncio
import logging
import torch
import numpy as np
from typing import Tuple, List

from app.config import settings

logger = logging.getLogger(__name__)


class VADService:
    """Voice Activity Detection service using Silero VAD."""
    
    def __init__(self):
        self.model = None
        self.utils = None
        self.is_initialized = False
        self.sample_rate = 16000
    
    async def initialize(self):
        """Initialize Silero VAD model."""
        if self.is_initialized:
            return
        
        if not settings.enable_vad:
            logger.info("VAD disabled in settings")
            return
        
        try:
            logger.info("Initializing Silero VAD...")
            
            loop = asyncio.get_event_loop()
            
            # Load model in executor to avoid blocking
            def load_model():
                model, utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                return model, utils
            
            self.model, self.utils = await loop.run_in_executor(None, load_model)
            
            self.is_initialized = True
            logger.info("✅ Silero VAD initialized successfully")
            
        except Exception as e:
            logger.warning(f"VAD initialization failed: {e}. Continuing without VAD.")
            self.is_initialized = False
    
    async def detect_speech(
        self,
        audio: np.ndarray,
        threshold: float = 0.5
    ) -> Tuple[bool, List[Tuple[int, int]]]:
        """
        Detect speech in audio.
        
        Args:
            audio: Audio data as numpy array (float32, normalized to [-1, 1])
            threshold: Speech probability threshold (0-1)
            
        Returns:
            Tuple of (has_speech, speech_segments)
            speech_segments: List of (start_sample, end_sample) tuples
        """
        if not self.is_initialized:
            # Return full audio as speech if VAD not available
            return True, [(0, len(audio))]
        
        try:
            loop = asyncio.get_event_loop()
            
            def detect():
                # Ensure audio is float32 and normalized
                if audio.dtype != np.float32:
                    audio_float = audio.astype(np.float32)
                else:
                    audio_float = audio.copy()
                
                # Normalize if needed
                max_val = np.max(np.abs(audio_float))
                if max_val > 1.0:
                    audio_float = audio_float / max_val
                
                # Convert to torch tensor
                audio_tensor = torch.from_numpy(audio_float)
                
                # Get speech timestamps
                speech_timestamps = self.utils[0](
                    audio_tensor,
                    self.model,
                    sampling_rate=self.sample_rate,
                    threshold=threshold,
                    min_speech_duration_ms=250,
                    min_silence_duration_ms=100,
                    window_size_samples=512,
                    speech_pad_ms=30
                )
                
                return speech_timestamps
            
            speech_timestamps = await loop.run_in_executor(None, detect)
            
            # Convert timestamps to segments
            segments = [(ts['start'], ts['end']) for ts in speech_timestamps]
            has_speech = len(segments) > 0
            
            return has_speech, segments
            
        except Exception as e:
            logger.warning(f"VAD detection failed: {e}. Returning full audio.")
            return True, [(0, len(audio))]
    
    async def trim_silence(
        self,
        audio: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Trim silence from audio, keeping only speech segments.
        
        Args:
            audio: Audio data as numpy array
            threshold: Speech probability threshold
            
        Returns:
            Trimmed audio with silence removed
        """
        if not self.is_initialized:
            return audio
        
        try:
            has_speech, segments = await self.detect_speech(audio, threshold)
            
            if not has_speech or not segments:
                logger.warning("No speech detected in audio")
                return audio
            
            # Concatenate all speech segments
            speech_parts = []
            for start, end in segments:
                speech_parts.append(audio[start:end])
            
            trimmed_audio = np.concatenate(speech_parts)
            
            # Log trimming stats
            original_duration = len(audio) / self.sample_rate
            trimmed_duration = len(trimmed_audio) / self.sample_rate
            saved = original_duration - trimmed_duration
            
            if saved > 0.1:  # Only log if significant
                logger.info(f"VAD trimmed {saved:.2f}s of silence ({saved/original_duration*100:.1f}%)")
            
            return trimmed_audio
            
        except Exception as e:
            logger.warning(f"VAD trimming failed: {e}. Returning original audio.")
            return audio
    
    async def get_speech_probability(self, audio: np.ndarray) -> float:
        """
        Get overall speech probability for audio.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Speech probability (0-1)
        """
        if not self.is_initialized:
            return 1.0
        
        try:
            loop = asyncio.get_event_loop()
            
            def get_prob():
                if audio.dtype != np.float32:
                    audio_float = audio.astype(np.float32)
                else:
                    audio_float = audio.copy()
                
                max_val = np.max(np.abs(audio_float))
                if max_val > 1.0:
                    audio_float = audio_float / max_val
                
                audio_tensor = torch.from_numpy(audio_float)
                speech_prob = self.model(audio_tensor, self.sample_rate).item()
                return speech_prob
            
            return await loop.run_in_executor(None, get_prob)
            
        except Exception as e:
            logger.warning(f"Speech probability calculation failed: {e}")
            return 1.0


# Global VAD service instance
vad_service = VADService()


async def initialize_vad() -> bool:
    """Initialize VAD service."""
    try:
        await vad_service.initialize()
        return True
    except Exception as e:
        logger.error(f"VAD initialization failed: {e}")
        return False
