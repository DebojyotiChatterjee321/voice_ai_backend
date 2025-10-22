#!/usr/bin/env python3
"""
Voice Assistant AI Backend Server Startup Script
Simple script to start the FastAPI server with proper configuration.
"""

import sys
import os
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import and run the server
from app.main import run_server

if __name__ == "__main__":
    print("🚀 Voice Assistant AI Backend")
    print("=" * 50)
    print("Starting FastAPI server...")
    print("Press Ctrl+C to stop the server")
    print("=" * 50)
    
    try:
        port = 8000  # Assuming the port number is 8000, replace with actual port number
        print("🎉 Voice Assistant AI Backend started successfully!")
        print(f"📊 Server running on: http://localhost:{port}")
        print(f"🎤 Voice Assistant Frontend: http://localhost:{port}")
        print(f"📚 API Documentation: http://localhost:{port}/docs")
        print(f"🔌 WebSocket endpoint: ws://localhost:{port}/ws")
        print()
        print("🚀 Quick Start:")
        print(f"1. Open browser: http://localhost:{port}")
        print("2. Grant microphone permissions when prompted")
        print("3. Click 'Start Recording' and speak")
        print("4. Listen to AI responses!")
        print()
        print("💡 Press Ctrl+C to stop the server")
        print("=" * 60)
        run_server()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)
