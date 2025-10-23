"""
Conversational WebSocket endpoint for real-time AI voice conversations.
Provides continuous audio streaming with automatic turn-taking.
"""

import json
import logging
import asyncio
from typing import Optional, Dict, Any
from datetime import datetime
import base64

from fastapi import WebSocket, WebSocketDisconnect, APIRouter
from fastapi.routing import APIRoute

from app.services.conversational_handler import session_manager

logger = logging.getLogger(__name__)

# Create router for conversational endpoints
conversational_router = APIRouter(
    prefix="/conversational",
    tags=["Conversational AI"]
)


@conversational_router.websocket("/ws")
async def conversational_websocket(websocket: WebSocket):
    """
    WebSocket endpoint for conversational AI with real-time audio streaming.
    
    Flow:
    1. Client connects and starts call
    2. Client sends continuous audio chunks
    3. Server detects speech boundaries with VAD
    4. Server processes: STT → LLM → TTS
    5. Server streams TTS audio back to client
    6. Return to listening for next user input
    
    Message Types:
    
    Client -> Server:
    - start_call: Initialize conversation session
    - audio_chunk: Audio data chunk for processing
    - end_call: End conversation session
    
    Server -> Client:
    - call_started: Session created and ready
    - speech_detected: User speech detected
    - transcription: User speech transcribed
    - bot_response: Bot text response
    - audio_chunk: TTS audio chunk
    - bot_speaking_end: Bot finished speaking
    - listening: Ready for user input
    - turn_complete: Turn metrics
    - error: Error occurred
    """
    session_id: Optional[str] = None
    
    try:
        # Accept WebSocket connection
        await websocket.accept()
        logger.info("Conversational WebSocket connection established")
        
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_established",
            "message": "Connected to conversational AI",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # WebSocket send function for session
        async def websocket_send(message: Dict[str, Any]):
            """Send message through WebSocket."""
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending WebSocket message: {e}")
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                message_type = message.get("type", "unknown")
                
                # Handle different message types
                if message_type == "start_call":
                    # Start new conversational session
                    config = message.get("config", {})
                    
                    session_id = await session_manager.create_session(
                        websocket_send=websocket_send,
                        config=config
                    )
                    
                    logger.info(f"Started conversational session: {session_id}")
                
                elif message_type == "audio_chunk":
                    # Process audio chunk
                    if not session_id:
                        await websocket.send_json({
                            "type": "error",
                            "error": "No active session. Start a call first.",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue
                    
                    session = await session_manager.get_session(session_id)
                    if not session:
                        await websocket.send_json({
                            "type": "error",
                            "error": "Session not found",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        continue
                    
                    # Decode audio data
                    audio_data_b64 = message.get("data", "")
                    if audio_data_b64:
                        try:
                            audio_bytes = base64.b64decode(audio_data_b64)
                            await session.process_audio_chunk(audio_bytes)
                        except Exception as e:
                            logger.error(f"Error decoding audio data: {e}")
                            await websocket.send_json({
                                "type": "error",
                                "error": f"Invalid audio data: {str(e)}",
                                "timestamp": datetime.utcnow().isoformat()
                            })
                
                elif message_type == "end_call":
                    # End conversational session
                    if session_id:
                        stats = await session_manager.end_session(session_id)
                        
                        await websocket.send_json({
                            "type": "call_ended",
                            "session_id": session_id,
                            "stats": stats,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        logger.info(f"Ended conversational session: {session_id}")
                        session_id = None
                
                elif message_type == "ping":
                    # Respond to ping
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
                elif message_type == "get_stats":
                    # Get session statistics
                    if session_id:
                        session = await session_manager.get_session(session_id)
                        if session:
                            stats = session.get_stats()
                            await websocket.send_json({
                                "type": "stats",
                                "stats": stats,
                                "timestamp": datetime.utcnow().isoformat()
                            })
                
                else:
                    # Unknown message type
                    await websocket.send_json({
                        "type": "error",
                        "error": f"Unknown message type: {message_type}",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
            except WebSocketDisconnect:
                logger.info("Client disconnected")
                break
                
            except json.JSONDecodeError:
                await websocket.send_json({
                    "type": "error",
                    "error": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                })
                
            except Exception as e:
                logger.error(f"Error handling message: {e}")
                await websocket.send_json({
                    "type": "error",
                    "error": f"Message processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    
    finally:
        # Clean up session on disconnect
        if session_id:
            try:
                await session_manager.end_session(session_id)
                logger.info(f"Cleaned up session {session_id} on disconnect")
            except Exception as e:
                logger.error(f"Error cleaning up session: {e}")


# REST API endpoints for conversational features
@conversational_router.get("/sessions")
async def list_sessions():
    """Get list of active conversational sessions."""
    stats = session_manager.get_all_stats()
    return {
        "active_sessions": stats["total_sessions"],
        "sessions": stats["sessions"],
        "timestamp": datetime.utcnow().isoformat()
    }


@conversational_router.get("/sessions/{session_id}")
async def get_session_stats(session_id: str):
    """Get statistics for a specific session."""
    session = await session_manager.get_session(session_id)
    
    if not session:
        return {
            "error": "Session not found",
            "session_id": session_id
        }
    
    return {
        "session_id": session_id,
        "stats": session.get_stats(),
        "timestamp": datetime.utcnow().isoformat()
    }


@conversational_router.delete("/sessions/{session_id}")
async def end_session(session_id: str):
    """End a conversational session."""
    stats = await session_manager.end_session(session_id)
    
    if not stats:
        return {
            "error": "Session not found",
            "session_id": session_id
        }
    
    return {
        "message": "Session ended",
        "session_id": session_id,
        "final_stats": stats,
        "timestamp": datetime.utcnow().isoformat()
    }


@conversational_router.post("/sessions/cleanup")
async def cleanup_inactive_sessions(timeout_seconds: int = 300):
    """Clean up inactive sessions."""
    await session_manager.cleanup_inactive_sessions(timeout_seconds)
    
    return {
        "message": "Cleanup completed",
        "active_sessions": session_manager.get_all_stats()["total_sessions"],
        "timestamp": datetime.utcnow().isoformat()
    }


@conversational_router.get("/config")
async def get_conversational_config():
    """Get conversational AI configuration."""
    from app.config import settings
    
    return {
        "enabled": True,
        "vad": {
            "enabled": settings.enable_vad,
            "default_threshold": 0.5
        },
        "stt": {
            "provider": "groq" if settings.use_groq_stt else "openai",
            "model": settings.groq_stt_model if settings.use_groq_stt else settings.whisper_model
        },
        "llm": {
            "provider": "groq" if settings.use_groq_llm else "openai",
            "model": settings.groq_llm_model if settings.use_groq_llm else settings.openai_model
        },
        "tts": {
            "provider": "cartesia" if settings.use_cartesia_tts else "elevenlabs",
            "streaming": settings.enable_streaming
        },
        "audio": {
            "sample_rate": 16000,
            "chunk_duration_ms": 200,
            "format": "webm/opus"
        },
        "performance": {
            "target_latency_ms": 1000,
            "expected_stt_ms": "100-200",
            "expected_llm_ms": "200-400",
            "expected_tts_ms": "100-200"
        }
    }


@conversational_router.get("/test")
async def test_conversational_endpoint():
    """Test endpoint to verify conversational setup."""
    from app.config import settings
    from app.services.vad import vad_service
    
    # Check component readiness
    components = {
        "websocket_endpoint": True,
        "session_manager": True,
        "vad_service": vad_service.is_initialized,
        "groq_stt": settings.use_groq_stt and settings.groq_api_key != "your_groq_api_key_here",
        "groq_llm": settings.use_groq_llm and settings.groq_api_key != "your_groq_api_key_here",
        "cartesia_tts": settings.use_cartesia_tts and settings.cartesia_api_key != "your_cartesia_api_key_here",
    }
    
    all_ready = all(components.values())
    
    return {
        "status": "ready" if all_ready else "partial",
        "components": components,
        "websocket_url": "/conversational/ws",
        "instructions": {
            "connect": "Connect to ws://localhost:8000/conversational/ws",
            "start_call": "Send: {'type': 'start_call', 'config': {...}}",
            "send_audio": "Send: {'type': 'audio_chunk', 'data': '<base64_audio>'}",
            "end_call": "Send: {'type': 'end_call'}"
        },
        "active_sessions": session_manager.get_all_stats()["total_sessions"]
    }


# Background task for session cleanup
async def cleanup_task():
    """Background task to periodically clean up inactive sessions."""
    while True:
        try:
            await asyncio.sleep(60)  # Run every minute
            await session_manager.cleanup_inactive_sessions(timeout_seconds=300)
        except Exception as e:
            logger.error(f"Error in cleanup task: {e}")


# Start cleanup task on module import
# Note: This will be properly managed by the FastAPI lifespan in main.py
_cleanup_task: Optional[asyncio.Task] = None


async def start_cleanup_task():
    """Start the cleanup background task."""
    global _cleanup_task
    if _cleanup_task is None:
        _cleanup_task = asyncio.create_task(cleanup_task())
        logger.info("Session cleanup task started")


async def stop_cleanup_task():
    """Stop the cleanup background task."""
    global _cleanup_task
    if _cleanup_task:
        _cleanup_task.cancel()
        try:
            await _cleanup_task
        except asyncio.CancelledError:
            pass
        _cleanup_task = None
        logger.info("Session cleanup task stopped")
