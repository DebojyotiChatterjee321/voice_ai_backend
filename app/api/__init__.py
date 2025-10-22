"""
API package for Voice Assistant AI Backend.
Contains all API route modules.
"""

from .voice import router as voice_router

# Export all routers
__all__ = [
    "voice_router"
]