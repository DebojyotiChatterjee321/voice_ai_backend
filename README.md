# Voice AI Assistant Backend

A high-performance, real-time voice assistant backend built with FastAPI, WebSockets, and modern AI technologies. This backend powers conversational AI interactions with support for speech-to-text, text-to-speech, and natural language understanding.

## Key Features

- **Real-time Voice Processing**: Low-latency audio streaming with WebSockets
- **Multi-Provider AI Integration**: Support for multiple AI providers (OpenAI, Groq)
- **Modular Architecture**: Easily extensible service architecture
- **Database Integration**: PostgreSQL with SQLAlchemy ORM
- **Authentication & Security**: JWT-based authentication and secure WebSocket connections
- **Performance Optimized**: Built for high concurrency and low latency
- **Containerized Deployment**: Ready for Docker and Kubernetes

## Technology Stack

- **Framework**: FastAPI
- **Database**: PostgreSQL
- **AI/ML**: 
  - OpenAI Whisper for STT
  - Groq for ultra-fast LLM inference
  - ElevenLabs for high-quality TTS
- **Real-time**: WebSockets, WebRTC
- **Caching**: Redis
- **Containerization**: Docker
- **Monitoring**: Prometheus metrics

## Installation

1. **Prerequisites**
   - Python 3.12+
   - PostgreSQL 14+
   - Redis 7+
   - FFmpeg (for audio processing)

2. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/voice_ai_backend.git
   cd voice_ai_backend
   ```

3. **Set up environment variables**
   Copy `.env.example` to `.env` and update the values:
   ```bash
   cp .env.example .env
   ```

4. **Install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   pip install -r requirements.txt
   ```

5. **Database setup**
   ```bash
   # Run migrations
   alembic upgrade head
   
   # Seed initial data (optional)
   python -m app.db.seed_data
   ```

## Running the Server

```bash
# Development server with hot-reload
uvicorn app.main:app --reload

# Production server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

## API Documentation

Once the server is running, access the interactive API documentation:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

## WebSocket Endpoints

### Voice Chat
- `ws://localhost:8000/ws/voice` - Real-time voice chat with the AI assistant

### Conversational Interface
- `ws://localhost:8000/ws/conversation` - Text-based conversational interface

## Project Structure

```
voice_ai_backend/
├── app/
│   ├── api/                  # API endpoints and WebSocket handlers
│   ├── core/                 # Core configurations and utilities
│   ├── db/                   # Database models and migrations
│   ├── models/               # SQLAlchemy models
│   ├── schemas/              # Pydantic models for request/response validation
│   └── services/             # Business logic and AI service integrations
│       ├── llm.py            # Language model services
│       ├── stt.py            # Speech-to-text services
│       └── tts.py            # Text-to-speech services
├── tests/                    # Test suite
├── .env.example              # Example environment variables
├── alembic.ini               # Database migration configuration
├── requirements.txt          # Project dependencies
└── start_server.py           # Application entry point
```

## Testing

Run the test suite:
```bash
pytest
```

## Docker Deployment

1. Build the Docker image:
   ```bash
   docker build -t voice-ai-backend .
   ```

2. Run with Docker Compose:
   ```bash
   docker-compose up -d
   ```

## Monitoring

- **Prometheus Metrics**: Available at `/metrics`
- **Structured Logging**: JSON-formatted logs with context

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

For support, please open an issue in the GitHub repository.
