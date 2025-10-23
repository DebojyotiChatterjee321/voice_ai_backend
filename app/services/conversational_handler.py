"""
Conversational Session Handler for real-time AI voice conversations.
Manages audio streaming, VAD, STT, LLM, and TTS pipeline for each session.
"""

import asyncio
import logging
import time
import uuid
from typing import Optional, Callable, Dict, Any, List
from enum import Enum
from datetime import datetime
import numpy as np

from app.services.realtime_vad import RealtimeVADProcessor, SpeechState, create_realtime_vad
from app.services.streaming_tts_manager import StreamingTTSManager, TTSChunkEncoder
from app.services import transcribe_audio, generate_response
from app.config import settings

logger = logging.getLogger(__name__)


class ConversationState(Enum):
    """States for conversational flow."""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ERROR = "error"


class ConversationalSession:
    """
    Manages a single conversational AI session with real-time audio processing.
    Handles the complete flow: Audio → VAD → STT → LLM → TTS → Audio
    """
    
    def __init__(
        self,
        session_id: str,
        websocket_send: Callable,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize conversational session.
        
        Args:
            session_id: Unique session identifier
            websocket_send: Async function to send messages via WebSocket
            config: Optional configuration overrides
        """
        self.session_id = session_id
        self.websocket_send = websocket_send
        self.config = config or {}
        
        # State management
        self.state = ConversationState.IDLE
        self.is_active = False
        
        # Initialize components
        self.vad_processor = create_realtime_vad(
            sample_rate=16000,
            chunk_duration_ms=self.config.get("chunk_duration_ms", 200),
            speech_threshold=self.config.get("vad_threshold", 0.5),
            silence_duration_ms=self.config.get("silence_duration_ms", 1500),  # Increased to 1.5s to handle natural pauses
            min_speech_duration_ms=self.config.get("min_speech_duration_ms", 250)
        )
        
        self.tts_manager = StreamingTTSManager(
            use_cartesia=self.config.get("use_cartesia", settings.use_cartesia_tts)
        )
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.max_history_length = 20
        
        # Audio buffer for collecting chunks
        self.audio_buffer: List[np.ndarray] = []
        self.chunk_sequence = 0
        
        # Performance tracking
        self.turn_count = 0
        self.total_latency = 0.0
        self.latency_samples: List[float] = []
        
        # Created timestamp
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        
        logger.info(f"ConversationalSession created: {session_id}")
    
    async def start(self):
        """Start the conversational session."""
        if self.is_active:
            logger.warning(f"Session {self.session_id} already active")
            return
        
        self.is_active = True
        self.state = ConversationState.LISTENING
        
        # Send greeting
        await self._send_message({
            "type": "call_started",
            "session_id": self.session_id,
            "message": "Connected! How can I help you today?",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Generate and play greeting
        await self._speak_greeting()
        
        logger.info(f"Session {self.session_id} started")
    
    async def process_audio_chunk(self, audio_data: bytes):
        """
        Process incoming audio chunk.
        
        Args:
            audio_data: Raw audio bytes from client
        """
        if not self.is_active:
            logger.warning(f"Session {self.session_id} not active, ignoring audio")
            return
        
        self.last_activity = datetime.utcnow()
        self.chunk_sequence += 1
        
        try:
            # Convert bytes to numpy array
            audio_array = self._bytes_to_numpy(audio_data)
            
            # Process through VAD
            vad_state, speech_segment = await self.vad_processor.process_chunk(audio_array)
            
            # Debug logging
            if self.chunk_sequence % 10 == 0:  # Log every 10th chunk
                logger.debug(f"Chunk {self.chunk_sequence}: VAD state={vad_state.value}, audio_len={len(audio_array)}")
            
            # Handle state transitions
            if vad_state == SpeechState.SPEECH_START:
                logger.info(f"Speech started (chunk {self.chunk_sequence})")
                await self._on_speech_start()
            
            elif vad_state == SpeechState.SPEECH:
                # User is speaking
                if self.state == ConversationState.SPEAKING:
                    # User interrupted bot
                    await self._on_user_interrupt()
            
            elif vad_state == SpeechState.SPEECH_END and speech_segment is not None:
                # Complete speech segment detected
                logger.info(f"Speech ended (chunk {self.chunk_sequence}), segment length: {len(speech_segment)/16000:.2f}s")
                await self._on_speech_complete(speech_segment)
            
        except Exception as e:
            logger.error(f"Error processing audio chunk: {e}")
            await self._send_error(f"Audio processing error: {str(e)}")
    
    async def _on_speech_start(self):
        """Handle speech start detection."""
        if self.state == ConversationState.LISTENING:
            await self._send_message({
                "type": "speech_detected",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def _on_user_interrupt(self):
        """Handle user interruption during bot speech."""
        logger.info(f"User interrupted bot in session {self.session_id}")
        
        self.state = ConversationState.INTERRUPTED
        
        # Stop TTS playback
        await self.tts_manager.interrupt()
        
        # Notify client to stop playback
        await self._send_message({
            "type": "bot_interrupted",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Transition to listening
        self.state = ConversationState.LISTENING
    
    async def _on_speech_complete(self, speech_segment: np.ndarray):
        """
        Handle complete speech segment.
        
        Args:
            speech_segment: Complete audio segment of user speech
        """
        if len(speech_segment) == 0:
            logger.warning("Empty speech segment, ignoring")
            return
        
        turn_start = time.time()
        self.state = ConversationState.PROCESSING
        self.turn_count += 1
        
        logger.info(
            f"Processing turn {self.turn_count} in session {self.session_id}: "
            f"{len(speech_segment)/16000:.2f}s audio"
        )
        
        try:
            # Step 1: Speech-to-Text
            stt_start = time.time()
            await self._send_message({
                "type": "processing_stt",
                "message": "Transcribing speech...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            transcription_result = await transcribe_audio(speech_segment, language="en")
            stt_time = time.time() - stt_start
            
            if not transcription_result["success"]:
                raise Exception("Transcription failed")
            
            user_text = transcription_result["text"].strip()
            
            if not user_text:
                logger.warning("Empty transcription, returning to listening")
                self.state = ConversationState.LISTENING
                await self._send_message({"type": "listening"})
                return
            
            logger.info(f"STT ({stt_time*1000:.0f}ms): '{user_text}'")
            
            # Send transcription to client
            await self._send_message({
                "type": "transcription",
                "text": user_text,
                "is_final": True,
                "confidence": transcription_result.get("confidence", 1.0),
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Step 2: LLM Processing
            llm_start = time.time()
            await self._send_message({
                "type": "processing_llm",
                "message": "Generating response...",
                "timestamp": datetime.utcnow().isoformat()
            })
            
            response_result = await generate_response(
                user_text,
                context={"conversation_history": self.conversation_history[-10:]}
            )
            llm_time = time.time() - llm_start
            
            if not response_result["success"]:
                raise Exception("LLM processing failed")
            
            bot_text = response_result["response"].strip()
            logger.info(f"LLM ({llm_time*1000:.0f}ms): '{bot_text[:50]}...'")
            
            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": bot_text,
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Trim history if too long
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Step 3: Text-to-Speech with streaming
            tts_start = time.time()
            self.state = ConversationState.SPEAKING
            
            sequence = 0
            bot_response_sent = False
            
            async def audio_chunk_callback(audio_chunk: bytes, is_final: bool):
                """Callback to send TTS audio chunks to client."""
                nonlocal sequence, bot_response_sent
                
                logger.debug(f"TTS callback: chunk_size={len(audio_chunk)}, is_final={is_final}, sequence={sequence}")
                
                # Send bot response text with first audio chunk for synchronization
                if not bot_response_sent and len(audio_chunk) > 0:
                    await self._send_message({
                        "type": "bot_response",
                        "text": bot_text,
                        "intent": response_result.get("intent", "general"),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    bot_response_sent = True
                
                if len(audio_chunk) > 0:
                    encoded = TTSChunkEncoder.encode_chunk(audio_chunk, sequence, is_final)
                    await self._send_message(encoded)
                    logger.info(f"Sent audio chunk {sequence}: {len(audio_chunk)} bytes")
                    sequence += 1
                
                if is_final:
                    logger.info(f"TTS complete, sent {sequence} audio chunks")
                    await self._send_message({
                        "type": "bot_speaking_end",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            tts_result = await self.tts_manager.synthesize_streaming(
                bot_text,
                callback=audio_chunk_callback
            )
            
            tts_time = time.time() - tts_start
            
            # Calculate total turn latency
            turn_latency = time.time() - turn_start
            self._update_latency_stats(turn_latency)
            
            logger.info(
                f"✅ Turn {self.turn_count} completed: {turn_latency*1000:.0f}ms total "
                f"(STT: {stt_time*1000:.0f}ms, LLM: {llm_time*1000:.0f}ms, "
                f"TTS: {tts_time*1000:.0f}ms)"
            )
            
            # Send performance metrics
            await self._send_message({
                "type": "turn_complete",
                "turn_number": self.turn_count,
                "transcription": user_text,
                "response": bot_text,
                "latency_ms": turn_latency * 1000,
                "breakdown": {
                    "stt_ms": stt_time * 1000,
                    "llm_ms": llm_time * 1000,
                    "tts_ms": tts_time * 1000
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Return to listening if not interrupted
            if self.state == ConversationState.SPEAKING:
                self.state = ConversationState.LISTENING
                await self._send_message({
                    "type": "listening",
                    "timestamp": datetime.utcnow().isoformat()
                })
            
        except Exception as e:
            logger.error(f"Error processing speech segment: {e}")
            self.state = ConversationState.ERROR
            await self._send_error(f"Processing error: {str(e)}")
            
            # Return to listening
            self.state = ConversationState.LISTENING
            await self._send_message({"type": "listening"})
    
    async def _speak_greeting(self):
        """Generate and speak greeting message."""
        greeting = "Hello! I'm your AI assistant. How can I help you today?"
        
        self.state = ConversationState.SPEAKING
        
        try:
            sequence = 0
            
            async def audio_chunk_callback(audio_chunk: bytes, is_final: bool):
                nonlocal sequence
                if len(audio_chunk) > 0:
                    encoded = TTSChunkEncoder.encode_chunk(audio_chunk, sequence, is_final)
                    await self._send_message(encoded)
                    sequence += 1
                
                if is_final:
                    await self._send_message({
                        "type": "bot_speaking_end",
                        "timestamp": datetime.utcnow().isoformat()
                    })
            
            await self.tts_manager.synthesize_streaming(
                greeting,
                callback=audio_chunk_callback
            )
            
            # Return to listening
            self.state = ConversationState.LISTENING
            await self._send_message({
                "type": "listening",
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error(f"Error generating greeting: {e}")
            self.state = ConversationState.LISTENING
    
    async def stop(self):
        """Stop the conversational session."""
        self.is_active = False
        self.state = ConversationState.IDLE
        
        # Interrupt any ongoing TTS
        if self.tts_manager.is_speaking():
            await self.tts_manager.interrupt()
        
        # Reset components
        self.vad_processor.reset()
        self.tts_manager.reset()
        
        logger.info(f"Session {self.session_id} stopped")
    
    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """
        Convert audio bytes to numpy array.
        
        Args:
            audio_bytes: Raw audio bytes
            
        Returns:
            Numpy array of audio samples
        """
        # Ensure buffer size is a multiple of 2 (int16 = 2 bytes)
        if len(audio_bytes) % 2 != 0:
            # Pad with a zero byte if odd length
            audio_bytes = audio_bytes + b'\x00'
        
        # Assume 16-bit PCM
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        
        # Normalize to float32 [-1, 1]
        audio_array = audio_array.astype(np.float32) / 32768.0
        
        return audio_array
    
    def _update_latency_stats(self, latency: float):
        """Update latency statistics."""
        self.latency_samples.append(latency)
        if len(self.latency_samples) > 100:
            self.latency_samples = self.latency_samples[-100:]
        
        self.total_latency += latency
    
    async def _send_message(self, message: Dict[str, Any]):
        """
        Send message via WebSocket.
        
        Args:
            message: Message dictionary
        """
        try:
            await self.websocket_send(message)
        except Exception as e:
            logger.error(f"Error sending message: {e}")
    
    async def _send_error(self, error_message: str):
        """
        Send error message to client.
        
        Args:
            error_message: Error description
        """
        await self._send_message({
            "type": "error",
            "error": error_message,
            "timestamp": datetime.utcnow().isoformat()
        })
    
    def get_stats(self) -> Dict[str, Any]:
        """Get session statistics."""
        avg_latency = (
            sum(self.latency_samples) / len(self.latency_samples)
            if self.latency_samples else 0
        )
        
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "is_active": self.is_active,
            "turn_count": self.turn_count,
            "conversation_length": len(self.conversation_history),
            "avg_latency_ms": avg_latency * 1000,
            "latency_p95_ms": (
                np.percentile(self.latency_samples, 95) * 1000
                if self.latency_samples else 0
            ),
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "vad_stats": self.vad_processor.get_stats(),
            "tts_stats": self.tts_manager.get_stats()
        }


class SessionManager:
    """Manages multiple conversational sessions."""
    
    def __init__(self):
        self.sessions: Dict[str, ConversationalSession] = {}
        self.session_lock = asyncio.Lock()
        logger.info("SessionManager initialized")
    
    async def create_session(
        self,
        websocket_send: Callable,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new conversational session.
        
        Args:
            websocket_send: WebSocket send function
            config: Optional configuration
            
        Returns:
            Session ID
        """
        session_id = str(uuid.uuid4())
        
        async with self.session_lock:
            session = ConversationalSession(session_id, websocket_send, config)
            self.sessions[session_id] = session
            await session.start()
        
        logger.info(f"Created session {session_id}. Total sessions: {len(self.sessions)}")
        return session_id
    
    async def get_session(self, session_id: str) -> Optional[ConversationalSession]:
        """Get session by ID."""
        return self.sessions.get(session_id)
    
    async def end_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """
        End a session and return stats.
        
        Args:
            session_id: Session ID
            
        Returns:
            Session statistics or None
        """
        async with self.session_lock:
            session = self.sessions.pop(session_id, None)
            
            if session:
                await session.stop()
                stats = session.get_stats()
                logger.info(f"Ended session {session_id}. Total sessions: {len(self.sessions)}")
                return stats
        
        return None
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all sessions."""
        return {
            "total_sessions": len(self.sessions),
            "sessions": {
                sid: session.get_stats()
                for sid, session in self.sessions.items()
            }
        }
    
    async def cleanup_inactive_sessions(self, timeout_seconds: int = 300):
        """Clean up inactive sessions."""
        now = datetime.utcnow()
        to_remove = []
        
        for session_id, session in self.sessions.items():
            inactive_seconds = (now - session.last_activity).total_seconds()
            if inactive_seconds > timeout_seconds:
                to_remove.append(session_id)
        
        for session_id in to_remove:
            await self.end_session(session_id)
            logger.info(f"Cleaned up inactive session {session_id}")


# Global session manager instance
session_manager = SessionManager()
