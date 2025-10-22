"""
Services package for Voice Assistant AI Backend.
Contains STT, LLM, and TTS services for real-time processing.
"""

from .stt import (
    WhisperSTTService,
    stt_service,
    transcribe_audio,
    initialize_stt
)

from .llm import (
    OpenAILLMService,
    llm_service,
    generate_response,
    analyze_and_respond,
    initialize_llm
)

from .tts import (
    ElevenLabsTTSService,
    tts_service,
    synthesize_speech,
    synthesize_streaming,
    initialize_tts
)

# Export all services
__all__ = [
    # STT Service
    "WhisperSTTService",
    "stt_service", 
    "transcribe_audio",
    "initialize_stt",
    
    # LLM Service
    "OpenAILLMService",
    "llm_service",
    "generate_response",
    "analyze_and_respond", 
    "initialize_llm",
    
    # TTS Service
    "ElevenLabsTTSService",
    "tts_service",
    "synthesize_speech",
    "synthesize_streaming",
    "initialize_tts"
]


async def initialize_all_services():
    """Initialize all AI services."""
    import logging
    logger = logging.getLogger(__name__)
    
    services = []
    
    try:
        # Initialize STT
        logger.info("Initializing STT service...")
        stt_success = await initialize_stt()
        services.append(("STT", stt_success))
        
        # Initialize LLM
        logger.info("Initializing LLM service...")
        llm_success = await initialize_llm()
        services.append(("LLM", llm_success))
        
        # Initialize TTS
        logger.info("Initializing TTS service...")
        tts_success = await initialize_tts()
        services.append(("TTS", tts_success))
        
        # Report results
        success_count = sum(1 for _, success in services if success)
        total_count = len(services)
        
        logger.info(f"Services initialized: {success_count}/{total_count}")
        
        for service_name, success in services:
            status = "✅" if success else "❌"
            logger.info(f"  {status} {service_name}")
        
        return success_count == total_count
        
    except Exception as e:
        logger.error(f"Service initialization failed: {e}")
        return False


async def health_check_all_services():
    """Check health of all services."""
    import logging
    logger = logging.getLogger(__name__)
    
    results = {}
    
    try:
        # Check STT
        results["stt"] = await stt_service.health_check()
        
        # Check LLM  
        results["llm"] = await llm_service.health_check()
        
        # Check TTS
        results["tts"] = await tts_service.health_check()
        
        # Overall status
        all_healthy = all(
            result.get("status") == "healthy" 
            for result in results.values()
        )
        
        results["overall"] = {
            "status": "healthy" if all_healthy else "unhealthy",
            "services_count": len(results) - 1,
            "healthy_count": sum(
                1 for result in results.values() 
                if result.get("status") == "healthy"
            )
        }
        
        return results
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "overall": {
                "status": "unhealthy",
                "error": str(e)
            }
        }