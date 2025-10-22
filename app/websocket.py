"""
WebSocket server implementation for real-time communication.
Handles client connections, message broadcasting, and signaling for voice assistant.
"""

import json
import logging
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import asyncio
import uuid

from fastapi import WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.routing import APIRouter
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.models import Customer, Product, Order
from app.db.connection import get_session
from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# WebSocket router
websocket_router = APIRouter()
api_router = APIRouter(prefix="/api/v1", tags=["API"])


class ConnectionManager:
    """Manages WebSocket connections and message broadcasting."""
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: Dict[str, WebSocket] = {}
        # Connection metadata
        self.connection_info: Dict[str, Dict[str, Any]] = {}
        # Room-based connections for group messaging
        self.rooms: Dict[str, Set[str]] = {}
        
    async def connect(self, websocket: WebSocket, client_id: str = None) -> str:
        """Accept a WebSocket connection and assign client ID."""
        await websocket.accept()
        
        # Generate client ID if not provided
        if not client_id:
            client_id = f"client_{uuid.uuid4().hex[:8]}"
        
        # Store connection
        self.active_connections[client_id] = websocket
        self.connection_info[client_id] = {
            "connected_at": datetime.utcnow().isoformat(),
            "last_activity": datetime.utcnow().isoformat(),
            "rooms": set()
        }
        
        logger.info(f"Client {client_id} connected. Total connections: {len(self.active_connections)}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "client_id": client_id,
            "message": "Connected to Voice Assistant AI Backend",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
        
        # Broadcast connection event to other clients
        await self.broadcast_message({
            "type": "client_connected",
            "client_id": client_id,
            "total_connections": len(self.active_connections),
            "timestamp": datetime.utcnow().isoformat()
        }, exclude_client=client_id)
        
        return client_id
    
    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            # Remove from all rooms
            client_rooms = self.connection_info.get(client_id, {}).get("rooms", set())
            for room in client_rooms:
                await self.leave_room(client_id, room)
            
            # Remove connection
            del self.active_connections[client_id]
            del self.connection_info[client_id]
            
            logger.info(f"Client {client_id} disconnected. Total connections: {len(self.active_connections)}")
            
            # Broadcast disconnection event
            await self.broadcast_message({
                "type": "client_disconnected",
                "client_id": client_id,
                "total_connections": len(self.active_connections),
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def send_personal_message(self, message: Dict[str, Any], client_id: str):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                websocket = self.active_connections[client_id]
                await websocket.send_text(json.dumps(message))
                
                # Update last activity
                if client_id in self.connection_info:
                    self.connection_info[client_id]["last_activity"] = datetime.utcnow().isoformat()
                    
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                await self.disconnect(client_id)
    
    async def broadcast_message(self, message: Dict[str, Any], exclude_client: str = None):
        """Broadcast a message to all connected clients."""
        disconnected_clients = []
        
        for client_id, websocket in self.active_connections.items():
            if exclude_client and client_id == exclude_client:
                continue
                
            try:
                await websocket.send_text(json.dumps(message))
                
                # Update last activity
                if client_id in self.connection_info:
                    self.connection_info[client_id]["last_activity"] = datetime.utcnow().isoformat()
                    
            except Exception as e:
                logger.error(f"Error broadcasting to {client_id}: {e}")
                disconnected_clients.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected_clients:
            await self.disconnect(client_id)
    
    async def join_room(self, client_id: str, room: str):
        """Add a client to a room for group messaging."""
        if room not in self.rooms:
            self.rooms[room] = set()
        
        self.rooms[room].add(client_id)
        
        if client_id in self.connection_info:
            self.connection_info[client_id]["rooms"].add(room)
        
        logger.info(f"Client {client_id} joined room {room}")
        
        # Notify room members
        await self.send_room_message({
            "type": "user_joined_room",
            "client_id": client_id,
            "room": room,
            "timestamp": datetime.utcnow().isoformat()
        }, room, exclude_client=client_id)
    
    async def leave_room(self, client_id: str, room: str):
        """Remove a client from a room."""
        if room in self.rooms and client_id in self.rooms[room]:
            self.rooms[room].remove(client_id)
            
            # Clean up empty rooms
            if not self.rooms[room]:
                del self.rooms[room]
        
        if client_id in self.connection_info:
            self.connection_info[client_id]["rooms"].discard(room)
        
        logger.info(f"Client {client_id} left room {room}")
        
        # Notify remaining room members
        await self.send_room_message({
            "type": "user_left_room",
            "client_id": client_id,
            "room": room,
            "timestamp": datetime.utcnow().isoformat()
        }, room)
    
    async def send_room_message(self, message: Dict[str, Any], room: str, exclude_client: str = None):
        """Send a message to all clients in a specific room."""
        if room not in self.rooms:
            return
        
        for client_id in self.rooms[room]:
            if exclude_client and client_id == exclude_client:
                continue
            await self.send_personal_message(message, client_id)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get current connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "active_rooms": len(self.rooms),
            "connections": {
                client_id: {
                    "connected_at": info["connected_at"],
                    "last_activity": info["last_activity"],
                    "rooms": list(info["rooms"])
                }
                for client_id, info in self.connection_info.items()
            },
            "rooms": {
                room: list(clients) for room, clients in self.rooms.items()
            }
        }


# Global connection manager
manager = ConnectionManager()


@websocket_router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket, client_id: Optional[str] = None):
    """Main WebSocket endpoint for client connections."""
    assigned_client_id = None
    
    try:
        # Connect client
        assigned_client_id = await manager.connect(websocket, client_id)
        
        # Message handling loop
        while True:
            try:
                # Receive message from client
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Process different message types
                await handle_websocket_message(assigned_client_id, message)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await manager.send_personal_message({
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.utcnow().isoformat()
                }, assigned_client_id)
            except Exception as e:
                logger.error(f"Error handling message from {assigned_client_id}: {e}")
                await manager.send_personal_message({
                    "type": "error",
                    "message": f"Message processing error: {str(e)}",
                    "timestamp": datetime.utcnow().isoformat()
                }, assigned_client_id)
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        if assigned_client_id:
            await manager.disconnect(assigned_client_id)


async def handle_websocket_message(client_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages based on type."""
    message_type = message.get("type", "unknown")
    
    if message_type == "ping":
        # Respond to ping with pong
        await manager.send_personal_message({
            "type": "pong",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)
    
    elif message_type == "broadcast":
        # Broadcast message to all clients
        await manager.broadcast_message({
            "type": "broadcast",
            "from_client": client_id,
            "message": message.get("message", ""),
            "timestamp": datetime.utcnow().isoformat()
        }, exclude_client=client_id)
    
    elif message_type == "join_room":
        # Join a specific room
        room = message.get("room")
        if room:
            await manager.join_room(client_id, room)
            await manager.send_personal_message({
                "type": "joined_room",
                "room": room,
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    elif message_type == "leave_room":
        # Leave a specific room
        room = message.get("room")
        if room:
            await manager.leave_room(client_id, room)
            await manager.send_personal_message({
                "type": "left_room",
                "room": room,
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
    
    elif message_type == "room_message":
        # Send message to specific room
        room = message.get("room")
        if room:
            await manager.send_room_message({
                "type": "room_message",
                "from_client": client_id,
                "room": room,
                "message": message.get("message", ""),
                "timestamp": datetime.utcnow().isoformat()
            }, room, exclude_client=client_id)
    
    elif message_type == "voice_data":
        # Handle voice data for AI processing
        await handle_voice_data(client_id, message)
    
    elif message_type == "text_message":
        # Handle text message for AI processing
        await handle_text_message(client_id, message)
    
    else:
        # Unknown message type
        await manager.send_personal_message({
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)


async def handle_voice_data(client_id: str, message: Dict[str, Any]):
    """Handle voice data for AI processing through Pipecat pipeline."""
    try:
        # Import here to avoid circular imports
        from app.pipecat import voice_pipeline
        import base64
        
        # Get or create session for this client
        session_id = message.get("session_id")
        if not session_id:
            # Create WebSocket callback for this client
            async def websocket_callback(response_data: Dict[str, Any]):
                """Send pipeline response back to WebSocket client."""
                # Convert audio data to base64 for WebSocket transmission
                if "audio_data" in response_data:
                    response_data["audio_data"] = base64.b64encode(response_data["audio_data"]).decode()
                
                await manager.send_personal_message({
                    "type": "voice_response",
                    **response_data
                }, client_id)
            
            # Start new session with WebSocket callback
            session_id = await voice_pipeline.start_session(websocket_callback=websocket_callback)
            
            # Send session info back to client
            await manager.send_personal_message({
                "type": "session_created",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Process the voice data
        voice_data = message.get("data", "")
        data_type = message.get("data_type", "base64")  # base64 or raw
        
        # Decode audio data if base64
        if data_type == "base64" and voice_data:
            audio_bytes = base64.b64decode(voice_data)
        else:
            audio_bytes = voice_data.encode() if isinstance(voice_data, str) else voice_data
        
        if audio_bytes:
            # Process through Pipecat pipeline
            result = await voice_pipeline.process_audio(session_id, audio_bytes)
            
            # Processing is handled by the pipeline callback, no need to send additional message
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": "No voice data received",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            
    except Exception as e:
        logger.error(f"Voice data processing error for client {client_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": f"Voice processing error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)


async def handle_text_message(client_id: str, message: Dict[str, Any]):
    """Handle text message for AI processing through Pipecat pipeline."""
    try:
        # Import here to avoid circular imports
        from app.pipecat import voice_pipeline
        import base64
        
        # Get or create session for this client
        session_id = message.get("session_id")
        if not session_id:
            # Create WebSocket callback for this client
            async def websocket_callback(response_data: Dict[str, Any]):
                """Send pipeline response back to WebSocket client."""
                # Convert audio data to base64 for WebSocket transmission
                if "audio_data" in response_data:
                    response_data["audio_data"] = base64.b64encode(response_data["audio_data"]).decode()
                
                await manager.send_personal_message({
                    "type": "text_response",
                    **response_data
                }, client_id)
            
            # Start new session with WebSocket callback
            session_id = await voice_pipeline.start_session(websocket_callback=websocket_callback)
            
            # Send session info back to client
            await manager.send_personal_message({
                "type": "session_created",
                "session_id": session_id,
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        
        # Process the text message
        text_content = message.get("text", "")
        
        if text_content:
            # Process through Pipecat pipeline
            result = await voice_pipeline.process_text(session_id, text_content)
            
            # Send processing confirmation
            await manager.send_personal_message({
                "type": "text_processing",
                "session_id": session_id,
                "status": "processing" if result["success"] else "error",
                "message": "Text message received and processing..." if result["success"] else result.get("error", "Processing failed"),
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
        else:
            await manager.send_personal_message({
                "type": "error",
                "message": "No text content received",
                "timestamp": datetime.utcnow().isoformat()
            }, client_id)
            
    except Exception as e:
        logger.error(f"Text message processing error for client {client_id}: {e}")
        await manager.send_personal_message({
            "type": "error",
            "message": f"Text processing error: {str(e)}",
            "timestamp": datetime.utcnow().isoformat()
        }, client_id)


# REST API Endpoints

@api_router.get("/customers")
async def list_customers(
    limit: int = 10,
    offset: int = 0,
    session: AsyncSession = Depends(get_session)
):
    """Get list of customers."""
    try:
        query = select(Customer).offset(offset).limit(limit)
        result = await session.execute(query)
        customers = result.scalars().all()
        
        return {
            "customers": [customer.to_dict() for customer in customers],
            "total": len(customers),
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error fetching customers: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch customers")


@api_router.get("/products")
async def list_products(
    limit: int = 10,
    offset: int = 0,
    category: Optional[str] = None,
    in_stock_only: bool = False,
    session: AsyncSession = Depends(get_session)
):
    """Get list of products with optional filtering."""
    try:
        query = select(Product)
        
        # Apply filters
        if category:
            query = query.where(Product.category == category)
        if in_stock_only:
            query = query.where(Product.stock_quantity > 0)
        
        query = query.offset(offset).limit(limit)
        result = await session.execute(query)
        products = result.scalars().all()
        
        return {
            "products": [product.to_dict() for product in products],
            "total": len(products),
            "limit": limit,
            "offset": offset,
            "filters": {
                "category": category,
                "in_stock_only": in_stock_only
            }
        }
    except Exception as e:
        logger.error(f"Error fetching products: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch products")


@api_router.get("/orders")
async def list_orders(
    limit: int = 10,
    offset: int = 0,
    customer_id: Optional[str] = None,
    session: AsyncSession = Depends(get_session)
):
    """Get list of orders with optional customer filtering."""
    try:
        query = select(Order)
        
        if customer_id:
            query = query.where(Order.customer_id == customer_id)
        
        query = query.offset(offset).limit(limit)
        result = await session.execute(query)
        orders = result.scalars().all()
        
        return {
            "orders": [order.to_dict() for order in orders],
            "total": len(orders),
            "limit": limit,
            "offset": offset,
            "filters": {
                "customer_id": customer_id
            }
        }
    except Exception as e:
        logger.error(f"Error fetching orders: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch orders")


@api_router.get("/websocket/stats")
async def websocket_stats():
    """Get WebSocket connection statistics."""
    return manager.get_connection_stats()


@api_router.post("/websocket/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all connected WebSocket clients."""
    try:
        await manager.broadcast_message({
            "type": "admin_broadcast",
            "message": message.get("message", ""),
            "timestamp": datetime.utcnow().isoformat()
        })
        
        return {
            "status": "success",
            "message": "Message broadcasted to all clients",
            "total_clients": len(manager.active_connections)
        }
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Failed to broadcast message")


@api_router.get("/websocket/test")
async def websocket_test():
    """Test endpoint to verify WebSocket functionality."""
    return {
        "websocket_url": f"ws://localhost:{settings.port}/ws",
        "test_instructions": {
            "connect": "Connect to the WebSocket endpoint",
            "ping": "Send: {'type': 'ping'}",
            "broadcast": "Send: {'type': 'broadcast', 'message': 'Hello everyone!'}",
            "join_room": "Send: {'type': 'join_room', 'room': 'test_room'}",
            "room_message": "Send: {'type': 'room_message', 'room': 'test_room', 'message': 'Hello room!'}"
        },
        "current_connections": len(manager.active_connections),
        "active_rooms": len(manager.rooms)
    }