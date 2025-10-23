"""
Pipecat pipeline integration for real-time voice assistant processing.
Orchestrates STT -> LLM -> TTS pipeline with sub-500ms latency.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, Callable, AsyncGenerator, List
import json
import uuid
from datetime import datetime

from pipecat.frames.frames import (
    AudioRawFrame,
    TextFrame,
    TTSAudioRawFrame,
    Frame,
    EndFrame,
    StartFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.ai_services import AIService

from app.services import (
    stt_service,
    llm_service, 
    tts_service,
    initialize_all_services
)
from app.services.llm import analyze_and_respond as llm_analyze_and_respond
from app.config import settings

logger = logging.getLogger(__name__)


class VoiceAssistantProcessor(FrameProcessor):
    """Custom Pipecat processor for voice assistant pipeline."""
    
    def __init__(self, session_id: str = None, websocket_callback: Callable = None):
        super().__init__()
        self.session_id = session_id or str(uuid.uuid4())
        self.conversation_history = []
        self.processing_stats = {
            "total_requests": 0,
            "avg_latency": 0.0,
            "last_processing_time": 0.0
        }
        # WebSocket callback for real-time communication
        self.websocket_callback = websocket_callback
        self.output_frames = []  # Store output frames for retrieval
        self._started = False  # Track if processor has been started
        
    async def process_frame(self, frame: Frame, direction: FrameDirection):
        """Process frames through the STT -> LLM -> TTS pipeline."""
        try:
            if isinstance(frame, StartFrame):
                # Mark processor as started and pass through the StartFrame
                self._started = True
                logger.info(f"VoiceAssistantProcessor started for session {self.session_id}")
                
            elif isinstance(frame, AudioRawFrame):
                # Audio input - process through STT
                await self._process_audio_input(frame, direction)
                
            elif isinstance(frame, TextFrame):
                # Text input - process through LLM -> TTS
                await self._process_text_input(frame, direction)
                
            elif isinstance(frame, EndFrame):
                # Handle end frame
                logger.info(f"VoiceAssistantProcessor ending for session {self.session_id}")
                
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            # Send error through WebSocket instead of pushing frame
            if self.websocket_callback:
                error_metadata = {
                    "type": "error",
                    "session_id": self.session_id,
                    "error": f"Processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.websocket_callback(error_metadata)
    
    async def _process_audio_input(self, frame: AudioRawFrame, direction: FrameDirection):
        """Process audio through STT -> LLM -> TTS pipeline."""
        start_time = time.time()
        
        try:
            # Step 1: Speech-to-Text
            stt_start = time.time()
            audio_data = frame.audio
            
            stt_result = await stt_service.transcribe_audio(audio_data)
            stt_time = time.time() - stt_start
            
            if not stt_result["success"] or not stt_result["text"].strip():
                logger.warning("STT failed or empty result")
                return
            
            transcribed_text = stt_result["text"]
            logger.info(f"STT ({stt_time:.3f}s): {transcribed_text}")
            
            # Step 2: LLM Processing
            llm_start = time.time()
            
            # Analyze intent and get context
            response_result = await llm_analyze_and_respond(transcribed_text)
            llm_time = time.time() - llm_start
            
            if not response_result["success"]:
                logger.error("LLM processing failed")
                return
            
            response_text = response_result["response"]
            logger.info(f"LLM ({llm_time:.3f}s): {response_text}")
            
            # Update conversation history
            self.conversation_history.extend([
                {"role": "user", "content": transcribed_text},
                {"role": "assistant", "content": response_text}
            ])
            
            # Keep history manageable
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Step 3: Text-to-Speech
            tts_start = time.time()
            
            tts_result = await tts_service.synthesize_speech(response_text)
            tts_time = time.time() - tts_start
            
            if not tts_result["success"]:
                logger.error("TTS processing failed")
                return
            
            logger.info(f"TTS ({tts_time:.3f}s): Generated {tts_result['audio_size']} bytes")
            
            # Create audio response frame
            audio_frame = TTSAudioRawFrame(
                audio=tts_result["audio_data"],
                sample_rate=44100,  # ElevenLabs default
                num_channels=1
            )
            
            # Calculate total processing time
            total_time = time.time() - start_time
            self._update_stats(total_time)
            
            # Log performance with optimization markers
            cached_marker = " [CACHED]" if response_result.get("cached", False) else ""
            logger.info(f"✅ OPTIMIZED Pipeline latency: {total_time:.3f}s (STT: {stt_time:.3f}s, LLM: {llm_time:.3f}s{cached_marker}, TTS: {tts_time:.3f}s)")
            logger.info(f"📊 Performance: {total_time*1000:.0f}ms total (Target: <2000ms)")
            
            # Store output frames for retrieval
            self.output_frames.append(audio_frame)
            
            # Create processing metadata
            metadata = {
                "type": "processing_complete",
                "session_id": self.session_id,
                "transcribed_text": transcribed_text,
                "response_text": response_text,
                "processing_time": total_time,
                "breakdown": {
                    "stt": stt_time,
                    "llm": llm_time,
                    "tts": tts_time
                },
                "intent": response_result.get("intent", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
                "audio_data": tts_result["audio_data"]  # Include audio for WebSocket
            }
            
            # Send to WebSocket if callback available
            if self.websocket_callback:
                await self.websocket_callback(metadata)
            
        except Exception as e:
            logger.error(f"Audio processing pipeline error: {e}")
            # Send error through WebSocket instead of pushing frame
            if self.websocket_callback:
                error_metadata = {
                    "type": "error",
                    "session_id": self.session_id,
                    "error": f"Sorry, I encountered an error processing your request: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.websocket_callback(error_metadata)
    
    async def _process_text_input(self, frame: TextFrame, direction: FrameDirection):
        """Process text input through LLM -> TTS pipeline."""
        start_time = time.time()
        
        try:
            text_input = frame.text
            
            # Skip metadata frames
            if text_input.startswith('{"type":'):
                await self.push_frame(frame, direction)
                return
            
            logger.info(f"Processing text input: {text_input}")
            
            # Step 1: LLM Processing
            llm_start = time.time()
            
            response_result = await llm_analyze_and_respond(text_input)
            llm_time = time.time() - llm_start
            
            if not response_result["success"]:
                logger.error("LLM processing failed")
                return
            
            response_text = response_result["response"]
            logger.info(f"LLM ({llm_time:.3f}s): {response_text}")
            
            # Step 2: Text-to-Speech
            tts_start = time.time()
            
            tts_result = await tts_service.synthesize_speech(response_text)
            tts_time = time.time() - tts_start
            
            if not tts_result["success"]:
                logger.error("TTS processing failed")
                return
            
            logger.info(f"TTS ({tts_time:.3f}s): Generated {tts_result['audio_size']} bytes")
            
            # Create audio response frame
            audio_frame = TTSAudioRawFrame(
                audio=tts_result["audio_data"],
                sample_rate=44100,
                num_channels=1
            )
            
            total_time = time.time() - start_time
            self._update_stats(total_time)
            
            logger.info(f"Text pipeline latency: {total_time:.3f}s (LLM: {llm_time:.3f}s, TTS: {tts_time:.3f}s)")
            
            # Store output frame instead of pushing it
            self.output_frames.append(audio_frame)
            
            # Send to WebSocket if callback available
            if self.websocket_callback:
                metadata = {
                    "type": "text_processing_complete",
                    "session_id": self.session_id,
                    "input_text": text_input,
                    "response_text": response_text,
                    "processing_time": total_time,
                    "breakdown": {
                        "llm": llm_time,
                        "tts": tts_time
                    },
                    "timestamp": datetime.utcnow().isoformat(),
                    "audio_data": tts_result["audio_data"]
                }
                await self.websocket_callback(metadata)
            
        except Exception as e:
            logger.error(f"Text processing pipeline error: {e}")
            # Send error through WebSocket instead of pushing frame
            if self.websocket_callback:
                error_metadata = {
                    "type": "error",
                    "session_id": self.session_id,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                await self.websocket_callback(error_metadata)
    
    def _update_stats(self, processing_time: float):
        """Update processing statistics."""
        self.processing_stats["total_requests"] += 1
        self.processing_stats["last_processing_time"] = processing_time
        
        # Update rolling average
        total_requests = self.processing_stats["total_requests"]
        current_avg = self.processing_stats["avg_latency"]
        
        self.processing_stats["avg_latency"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get processing statistics."""
        return {
            "session_id": self.session_id,
            "conversation_length": len(self.conversation_history),
            "output_frames_count": len(self.output_frames),
            **self.processing_stats
        }
    
    def get_latest_output_frame(self) -> Optional[Frame]:
        """Get the most recent output frame."""
        return self.output_frames[-1] if self.output_frames else None
    
    def clear_output_frames(self):
        """Clear stored output frames to free memory."""
        self.output_frames.clear()
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.conversation_history.copy()


class VoiceAssistantPipeline:
    """Main voice assistant pipeline using Pipecat."""
    
    def __init__(self):
        self.pipeline = None
        self.runner = None
        self.processor = None
        self.is_running = False
        self.session_callbacks = {}
        
    async def initialize(self):
        """Initialize the pipeline."""
        try:
            logger.info("Initializing Voice Assistant Pipeline")
            
            # Initialize all AI services
            services_ready = await initialize_all_services()
            if not services_ready:
                raise RuntimeError("Failed to initialize AI services")
            
            # Create processor
            self.processor = VoiceAssistantProcessor()
            
            # Create pipeline
            self.pipeline = Pipeline([self.processor])
            
            # Create pipeline runner (don't start yet - will be started per session)
            self.runner = PipelineRunner()
            self.task = None  # Will be created per session
            
            logger.info("Voice Assistant Pipeline initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline initialization failed: {e}")
            return False
    
    async def start_session(self, session_id: str = None, websocket_callback: Callable = None) -> str:
        """Start a new voice assistant session with optional WebSocket callback."""
        try:
            session_id = session_id or str(uuid.uuid4())
            
            # Create session-specific processor with WebSocket callback
            session_processor = VoiceAssistantProcessor(session_id, websocket_callback)
            
            # Store session
            self.session_callbacks[session_id] = session_processor
            
            logger.info(f"Started voice assistant session: {session_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start session: {e}")
            raise
    
    async def process_audio(self, session_id: str, audio_data: bytes) -> Dict[str, Any]:
        """Process audio input for a session."""
        try:
            if session_id not in self.session_callbacks:
                raise ValueError(f"Session {session_id} not found")
            
            processor = self.session_callbacks[session_id]
            
            # Send StartFrame if this is the first audio frame
            if not processor._started:
                start_frame = StartFrame()
                await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
            
            # Create audio frame
            audio_frame = AudioRawFrame(
                audio=audio_data,
                sample_rate=16000,  # Whisper default
                num_channels=1
            )
            
            # Process through pipeline
            await processor.process_frame(audio_frame, FrameDirection.DOWNSTREAM)
            
            return {
                "success": True,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Audio processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def process_text(self, session_id: str, text: str) -> Dict[str, Any]:
        """Process text input for a session."""
        try:
            if session_id not in self.session_callbacks:
                raise ValueError(f"Session {session_id} not found")
            
            processor = self.session_callbacks[session_id]
            
            # Send StartFrame if this is the first frame
            if not processor._started:
                start_frame = StartFrame()
                await processor.process_frame(start_frame, FrameDirection.DOWNSTREAM)
            
            # Create text frame
            text_frame = TextFrame(text)
            
            # Process through pipeline
            await processor.process_frame(text_frame, FrameDirection.DOWNSTREAM)
            
            return {
                "success": True,
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Text processing failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "session_id": session_id
            }
    
    async def end_session(self, session_id: str):
        """End a voice assistant session."""
        try:
            if session_id in self.session_callbacks:
                processor = self.session_callbacks[session_id]
                stats = processor.get_stats()
                
                del self.session_callbacks[session_id]
                
                logger.info(f"Ended session {session_id}. Stats: {stats}")
                return stats
            
        except Exception as e:
            logger.error(f"Failed to end session: {e}")
    
    async def get_session_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics for a session."""
        if session_id in self.session_callbacks:
            return self.session_callbacks[session_id].get_stats()
        return {}
    
    async def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        return {
            "active_sessions": len(self.session_callbacks),
            "sessions": {
                sid: processor.get_stats() 
                for sid, processor in self.session_callbacks.items()
            }
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Check pipeline health."""
        try:
            # Check AI services
            from app.services import health_check_all_services
            services_health = await health_check_all_services()
            
            return {
                "pipeline_status": "healthy" if self.processor else "not_initialized",
                "active_sessions": len(self.session_callbacks),
                "services": services_health,
                "overall_healthy": (
                    self.processor is not None and 
                    services_health.get("overall", {}).get("status") == "healthy"
                )
            }
            
        except Exception as e:
            return {
                "pipeline_status": "unhealthy",
                "error": str(e)
            }


# Global pipeline instance
voice_pipeline = VoiceAssistantPipeline()


async def initialize_pipeline() -> bool:
    """Initialize the voice assistant pipeline."""
    return await voice_pipeline.initialize()


async def create_voice_session() -> str:
    """Create a new voice assistant session."""
    return await voice_pipeline.start_session()


async def process_voice_input(session_id: str, audio_data: bytes) -> Dict[str, Any]:
    """Process voice input through the pipeline."""
    return await voice_pipeline.process_audio(session_id, audio_data)


async def process_text_input(session_id: str, text: str) -> Dict[str, Any]:
    """Process text input through the pipeline."""
    return await voice_pipeline.process_text(session_id, text)