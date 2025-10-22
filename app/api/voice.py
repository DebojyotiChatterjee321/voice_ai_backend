"""
Voice Assistant API endpoints for STT, LLM, and TTS processing.
"""

import logging
from typing import Dict, Any, Optional
import base64
import io

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Depends
from fastapi.responses import Response
from pydantic import BaseModel

from app.services import (
    transcribe_audio,
    generate_response,
    analyze_and_respond,
    synthesize_speech,
    synthesize_streaming
)
from app.pipecat import (
    voice_pipeline,
    create_voice_session,
    process_voice_input,
    process_text_input
)

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/api/v1/voice", tags=["Voice Assistant"])


# Request/Response models
class TextToSpeechRequest(BaseModel):
    text: str
    voice_id: Optional[str] = None
    model: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    context: Optional[Dict[str, Any]] = None


class VoiceSessionRequest(BaseModel):
    session_id: Optional[str] = None


class AudioProcessRequest(BaseModel):
    session_id: str
    audio_base64: str


# STT Endpoints
@router.post("/stt")
async def speech_to_text(
    audio_file: UploadFile = File(...),
    language: Optional[str] = Form("en")
):
    """Convert speech to text using Whisper."""
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Transcribe
        result = await transcribe_audio(audio_data, language)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Transcription failed"))
        
        return {
            "text": result["text"],
            "language": result.get("language", language),
            "confidence": result.get("confidence", 0.0),
            "processing_time": result.get("processing_time", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text processing failed")


@router.post("/stt/base64")
async def speech_to_text_base64(
    audio_base64: str = Form(...),
    language: Optional[str] = Form("en")
):
    """Convert speech to text from base64 encoded audio."""
    try:
        # Decode base64 audio
        audio_data = base64.b64decode(audio_base64)
        
        # Transcribe
        result = await transcribe_audio(audio_data, language)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Transcription failed"))
        
        return {
            "text": result["text"],
            "language": result.get("language", language),
            "confidence": result.get("confidence", 0.0),
            "processing_time": result.get("processing_time", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"STT base64 endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Speech-to-text processing failed")


# LLM Endpoints
@router.post("/chat")
async def chat_completion(request: ChatRequest):
    """Generate response using LLM with context analysis."""
    try:
        result = await analyze_and_respond(request.message)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Chat completion failed"))
        
        return {
            "response": result["response"],
            "intent": result.get("intent", "unknown"),
            "entities": result.get("entities", {}),
            "context": result.get("context", {}),
            "processing_time": result.get("processing_time", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Chat completion failed")


@router.post("/chat/simple")
async def simple_chat(
    message: str = Form(...),
    context: Optional[str] = Form(None)
):
    """Simple chat endpoint for quick responses."""
    try:
        # Parse context if provided
        context_dict = None
        if context:
            import json
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError:
                pass
        
        result = await generate_response(message, context_dict)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Response generation failed"))
        
        return {
            "response": result["response"],
            "processing_time": result.get("processing_time", 0.0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Simple chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Chat processing failed")


# TTS Endpoints
@router.post("/tts")
async def text_to_speech(request: TextToSpeechRequest):
    """Convert text to speech using ElevenLabs."""
    try:
        result = await synthesize_speech(
            request.text,
            request.voice_id,
            request.model
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Speech synthesis failed"))
        
        return {
            "audio_base64": result["audio_base64"],
            "format": result.get("format", "mp3"),
            "voice_id": result.get("voice_id"),
            "processing_time": result.get("processing_time", 0.0),
            "audio_size": result.get("audio_size", 0)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Text-to-speech processing failed")


@router.post("/tts/audio")
async def text_to_speech_audio(request: TextToSpeechRequest):
    """Convert text to speech and return audio file."""
    try:
        result = await synthesize_speech(
            request.text,
            request.voice_id,
            request.model
        )
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Speech synthesis failed"))
        
        # Return audio as response
        return Response(
            content=result["audio_data"],
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "attachment; filename=speech.mp3",
                "X-Processing-Time": str(result.get("processing_time", 0.0)),
                "X-Audio-Size": str(result.get("audio_size", 0))
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"TTS audio endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Text-to-speech processing failed")


# Pipeline Endpoints
@router.post("/session/create")
async def create_session():
    """Create a new voice assistant session."""
    try:
        session_id = await create_voice_session()
        
        return {
            "session_id": session_id,
            "status": "created",
            "message": "Voice assistant session created successfully"
        }
        
    except Exception as e:
        logger.error(f"Session creation error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create session")


@router.post("/session/{session_id}/process-audio")
async def process_session_audio(
    session_id: str,
    audio_file: UploadFile = File(...)
):
    """Process audio through the complete STT->LLM->TTS pipeline."""
    try:
        if not audio_file.content_type.startswith('audio/'):
            raise HTTPException(status_code=400, detail="File must be an audio file")
        
        # Read audio data
        audio_data = await audio_file.read()
        
        # Process through pipeline
        result = await process_voice_input(session_id, audio_data)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Audio processing failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session audio processing error: {e}")
        raise HTTPException(status_code=500, detail="Audio processing failed")


@router.post("/session/{session_id}/process-text")
async def process_session_text(
    session_id: str,
    message: str = Form(...)
):
    """Process text through the LLM->TTS pipeline."""
    try:
        result = await process_text_input(session_id, message)
        
        if not result["success"]:
            raise HTTPException(status_code=500, detail=result.get("error", "Text processing failed"))
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session text processing error: {e}")
        raise HTTPException(status_code=500, detail="Text processing failed")


@router.get("/session/{session_id}/stats")
async def get_session_stats(session_id: str):
    """Get statistics for a voice assistant session."""
    try:
        stats = await voice_pipeline.get_session_stats(session_id)
        
        if not stats:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return stats
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session stats")


@router.delete("/session/{session_id}")
async def end_session(session_id: str):
    """End a voice assistant session."""
    try:
        stats = await voice_pipeline.end_session(session_id)
        
        return {
            "session_id": session_id,
            "status": "ended",
            "final_stats": stats
        }
        
    except Exception as e:
        logger.error(f"Session end error: {e}")
        raise HTTPException(status_code=500, detail="Failed to end session")


@router.get("/pipeline/stats")
async def get_pipeline_stats():
    """Get overall pipeline statistics."""
    try:
        stats = await voice_pipeline.get_all_stats()
        return stats
        
    except Exception as e:
        logger.error(f"Pipeline stats error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get pipeline stats")


@router.get("/pipeline/health")
async def pipeline_health():
    """Check voice pipeline health."""
    try:
        health = await voice_pipeline.health_check()
        
        status_code = 200 if health.get("overall_healthy", False) else 503
        
        return Response(
            content=health,
            status_code=status_code,
            media_type="application/json"
        )
        
    except Exception as e:
        logger.error(f"Pipeline health check error: {e}")
        raise HTTPException(status_code=500, detail="Health check failed")


# Utility endpoints
@router.get("/test")
async def test_voice_services():
    """Test all voice services with sample data."""
    try:
        import numpy as np
        
        results = {}
        
        # Test STT with silence
        test_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
        stt_result = await transcribe_audio(test_audio)
        results["stt"] = {
            "success": stt_result["success"],
            "processing_time": stt_result.get("processing_time", 0.0)
        }
        
        # Test LLM
        llm_result = await generate_response("Hello, this is a test.")
        results["llm"] = {
            "success": llm_result["success"],
            "processing_time": llm_result.get("processing_time", 0.0)
        }
        
        # Test TTS
        tts_result = await synthesize_speech("Hello, this is a test.")
        results["tts"] = {
            "success": tts_result["success"],
            "processing_time": tts_result.get("processing_time", 0.0),
            "audio_size": tts_result.get("audio_size", 0)
        }
        
        # Overall success
        all_success = all(result["success"] for result in results.values())
        total_time = sum(result["processing_time"] for result in results.values())
        
        return {
            "overall_success": all_success,
            "total_processing_time": total_time,
            "services": results,
            "latency_target": "< 0.5s",
            "meets_target": total_time < 0.5
        }
        
    except Exception as e:
        logger.error(f"Voice services test error: {e}")
        raise HTTPException(status_code=500, detail="Service test failed")
