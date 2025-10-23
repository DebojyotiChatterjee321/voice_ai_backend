"""
Voice Assistant AI Backend - FastAPI Application
E-commerce Customer Support Bot with Real-time AI Processing
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import uvicorn
import logging
from pathlib import Path
import logging
from datetime import datetime

from app.config import settings
from app.db.connection import connect_database, disconnect_database
from app.websocket import websocket_router, api_router
from app.pipecat import initialize_pipeline, voice_pipeline
from app.services import initialize_all_services, health_check_all_services
from app.api import voice_router
from app.api.conversational_websocket import conversational_router, start_cleanup_task, stop_cleanup_task

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    logger.info("🚀 Starting Voice Assistant AI Backend...")
    try:
        # Initialize database
        await connect_database()
        logger.info("✅ Database connected successfully")
        
        # Initialize AI services and pipeline
        logger.info("🤖 Initializing AI services...")
        services_ready = await initialize_all_services()
        if services_ready:
            logger.info("✅ AI services initialized successfully")
        else:
            logger.warning("⚠️ Some AI services failed to initialize")
        
        # Initialize Pipecat pipeline
        logger.info("🎙️ Initializing voice pipeline...")
        pipeline_ready = await initialize_pipeline()
        if pipeline_ready:
            logger.info("✅ Voice pipeline initialized successfully")
        else:
            logger.warning("⚠️ Voice pipeline initialization failed")
        
        # Start conversational session cleanup task
        logger.info("🔄 Starting conversational session cleanup task...")
        await start_cleanup_task()
        logger.info("✅ Cleanup task started")
            
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Shutting down Voice Assistant AI Backend...")
    try:
        # Stop cleanup task
        await stop_cleanup_task()
        logger.info("✅ Cleanup task stopped")
        
        await disconnect_database()
        logger.info("✅ Database disconnected successfully")
    except Exception as e:
        logger.error(f"❌ Database disconnection error: {e}")


# Create FastAPI application
app = FastAPI(
    title="Voice Assistant AI Backend",
    description="E-commerce Customer Support Bot with Real-time AI Processing using FastAPI and Pipecat",
    version=settings.version,
    debug=settings.debug,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
static_dir.mkdir(exist_ok=True)  # Create static directory if it doesn't exist
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Include routers
app.include_router(websocket_router)
app.include_router(api_router)
app.include_router(voice_router)
app.include_router(conversational_router)  # New conversational AI router


@app.get("/")
async def root():
    """Serve the voice assistant frontend."""
    return FileResponse(static_dir / "index.html")

@app.get("/api")
async def api_root():
    """API root endpoint - Basic health check."""
    return {
        "message": "🎤 Voice Assistant AI Backend is running!",
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.version,
        "docs_url": "/docs",
        "websocket_url": f"ws://localhost:{settings.port}/ws",
        "conversational_websocket_url": f"ws://localhost:{settings.port}/conversational/ws",
        "frontend_url": f"http://localhost:{settings.port}/"
    }


@app.get("/health")
async def health_check():
    """Detailed health check endpoint."""
    try:
        from app.db.connection import db_manager
        
        # Check database connectivity
        db_healthy = await db_manager.health_check() if db_manager.is_connected else False
        
        # Check AI services
        services_health = await health_check_all_services()
        
        # Check voice pipeline
        pipeline_health = await voice_pipeline.health_check()
        
        # Overall health status
        overall_healthy = (
            db_healthy and 
            services_health.get("overall", {}).get("status") == "healthy" and
            pipeline_health.get("overall_healthy", False)
        )
        
        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "timestamp": datetime.utcnow().isoformat(),
            "app_name": settings.app_name,
            "version": settings.version,
            "components": {
                "database": "connected" if db_healthy else "disconnected",
                "ai_services": services_health,
                "voice_pipeline": pipeline_health
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/info")
async def app_info():
    """Application information endpoint."""
    return {
        "app_name": settings.app_name,
        "version": settings.version,
        "debug": settings.debug,
        "environment": "development" if settings.debug else "production",
        "database_host": settings.database_host,
        "database_name": settings.database_name,
        "features": [
            "FastAPI",
            "PostgreSQL with AsyncPG",
            "Pipecat AI Integration",
            "Real-time WebSocket Support",
            "Conversational AI (VAD + Streaming)",
            "OpenAI Integration",
            "Whisper STT",
            "Groq Ultra-fast Processing",
            "Cartesia Streaming TTS",
            "E-commerce Customer Support"
        ],
        "endpoints": {
            "root": "/",
            "health": "/health",
            "info": "/info",
            "docs": "/docs",
            "redoc": "/redoc"
        }
    }


def run_server():
    """Run the FastAPI server with uvicorn."""
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
        access_log=True
    )


if __name__ == "__main__":
    print("🚀 Starting Voice Assistant AI Backend Server...")
    print(f"📍 Server will run on: http://{settings.host}:{settings.port}")
    print(f"📚 API Documentation: http://{settings.host}:{settings.port}/docs")
    print(f"🔧 Debug Mode: {settings.debug}")
    print("=" * 60)
    
    run_server()