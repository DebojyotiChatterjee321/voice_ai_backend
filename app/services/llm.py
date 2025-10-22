"""
Large Language Model (LLM) service using OpenAI GPT-4o-mini.
Optimized for low-latency customer support responses.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
import json

import openai
from openai import AsyncOpenAI
import httpx

from app.config import settings
from app.models import Customer, Product, Order
from app.db.connection import get_database

logger = logging.getLogger(__name__)


class OpenAILLMService:
    """Asynchronous OpenAI LLM service optimized for customer support."""
    
    def __init__(self):
        self.client = None
        self.model = settings.openai_model
        self.max_tokens = settings.openai_max_tokens
        self.is_initialized = False
        
        # Performance settings
        self.temperature = 0.1  # Low temperature for consistent responses
        self.timeout = 10.0  # 10 second timeout
        self.max_context_length = 4000  # Keep context short for speed
        
        # Customer support system prompt
        self.system_prompt = """You are a helpful customer support AI assistant for an e-commerce platform. 

Your role:
- Provide quick, concise, and helpful responses
- Help customers with order inquiries, product information, and general support
- Keep responses under 50 words when possible
- Be friendly but professional
- If you need specific order/customer data, ask for order ID or customer ID

Guidelines:
- Always be polite and empathetic
- Provide actionable solutions
- Escalate complex issues to human agents when needed
- Never make promises about refunds or exchanges without verification
- Focus on immediate help the customer needs

Response format: Keep it brief and direct."""
    
    async def initialize(self):
        """Initialize OpenAI client."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing OpenAI LLM service")
            
            # Initialize async client with timeout settings
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key,
                timeout=httpx.Timeout(self.timeout),
                max_retries=2
            )
            
            # Test connection with a simple request
            await self._test_connection()
            
            self.is_initialized = True
            logger.info("OpenAI LLM service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI LLM service: {e}")
            raise
    
    async def _test_connection(self):
        """Test OpenAI API connection."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            logger.info("OpenAI API connection test successful")
        except Exception as e:
            logger.error(f"OpenAI API connection test failed: {e}")
            raise
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response to user message.
        
        Args:
            user_message: User's input message
            context: Additional context (customer data, order info, etc.)
            conversation_history: Previous conversation messages
            
        Returns:
            Dictionary with response and metadata
        """
        if not self.is_initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Build messages
            messages = await self._build_messages(user_message, context, conversation_history)
            
            # Generate response
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=False,  # Disable streaming for lower latency
                presence_penalty=0.1,
                frequency_penalty=0.1
            )
            
            processing_time = time.time() - start_time
            
            # Extract response
            assistant_message = response.choices[0].message.content.strip()
            
            return {
                "response": assistant_message,
                "processing_time": processing_time,
                "model": self.model,
                "tokens_used": response.usage.total_tokens,
                "success": True,
                "finish_reason": response.choices[0].finish_reason
            }
            
        except Exception as e:
            logger.error(f"LLM response generation failed: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team.",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False
            }
    
    async def _build_messages(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Build message array for OpenAI API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context if provided
        if context:
            context_message = await self._format_context(context)
            if context_message:
                messages.append({"role": "system", "content": context_message})
        
        # Add conversation history (keep it short for speed)
        if conversation_history:
            # Only keep last 4 messages to maintain low latency
            recent_history = conversation_history[-4:]
            messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
        # Ensure total context doesn't exceed limit
        messages = await self._trim_context(messages)
        
        return messages
    
    async def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context information for the LLM."""
        context_parts = []
        
        if "customer" in context:
            customer = context["customer"]
            context_parts.append(f"Customer: {customer.get('name', 'Unknown')} (ID: {customer.get('customer_id', 'Unknown')})")
        
        if "order" in context:
            order = context["order"]
            context_parts.append(f"Order: {order.get('order_id', 'Unknown')} - Status: {order.get('status', 'Unknown')} - Amount: ${order.get('total_amount', 0)}")
        
        if "product" in context:
            product = context["product"]
            context_parts.append(f"Product: {product.get('name', 'Unknown')} - Category: {product.get('category', 'Unknown')} - Stock: {product.get('stock_quantity', 0)}")
        
        if context_parts:
            return "Context: " + " | ".join(context_parts)
        
        return ""
    
    async def _trim_context(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Trim context to stay within token limits."""
        # Rough estimation: 1 token ≈ 4 characters
        total_chars = sum(len(msg["content"]) for msg in messages)
        
        if total_chars > self.max_context_length:
            # Keep system message and last few user/assistant exchanges
            system_msgs = [msg for msg in messages if msg["role"] == "system"]
            other_msgs = [msg for msg in messages if msg["role"] != "system"]
            
            # Keep last few messages that fit in context
            trimmed_msgs = system_msgs.copy()
            char_count = sum(len(msg["content"]) for msg in system_msgs)
            
            for msg in reversed(other_msgs):
                msg_chars = len(msg["content"])
                if char_count + msg_chars <= self.max_context_length:
                    trimmed_msgs.insert(-len(system_msgs), msg)
                    char_count += msg_chars
                else:
                    break
            
            return trimmed_msgs
        
        return messages
    
    async def generate_streaming_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> AsyncGenerator[str, None]:
        """
        Generate streaming response for real-time applications.
        
        Args:
            user_message: User's input message
            context: Additional context
            
        Yields:
            Response chunks as they're generated
        """
        if not self.is_initialized:
            await self.initialize()
        
        try:
            messages = await self._build_messages(user_message, context, None)
            
            stream = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True
            )
            
            async for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            logger.error(f"Streaming response failed: {e}")
            yield "I apologize, but I'm experiencing technical difficulties."
    
    async def analyze_intent(self, user_message: str) -> Dict[str, Any]:
        """
        Analyze user intent for routing and context gathering.
        
        Args:
            user_message: User's input message
            
        Returns:
            Intent analysis results
        """
        try:
            intent_prompt = """Analyze the user's message and classify the intent. Respond with JSON only:

{
  "intent": "order_inquiry|product_question|general_support|complaint|compliment|other",
  "entities": {
    "order_id": "extracted order ID if mentioned",
    "product_name": "extracted product name if mentioned",
    "customer_id": "extracted customer ID if mentioned"
  },
  "urgency": "low|medium|high",
  "requires_human": true/false
}

User message: """ + user_message
            
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": intent_prompt}],
                max_tokens=200,
                temperature=0
            )
            
            # Parse JSON response
            result = json.loads(response.choices[0].message.content)
            return {
                "success": True,
                "intent": result.get("intent", "other"),
                "entities": result.get("entities", {}),
                "urgency": result.get("urgency", "medium"),
                "requires_human": result.get("requires_human", False)
            }
            
        except Exception as e:
            logger.error(f"Intent analysis failed: {e}")
            return {
                "success": False,
                "intent": "other",
                "entities": {},
                "urgency": "medium",
                "requires_human": False,
                "error": str(e)
            }
    
    async def get_context_from_database(self, entities: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fetch relevant context from database based on extracted entities.
        
        Args:
            entities: Extracted entities from intent analysis
            
        Returns:
            Context data from database
        """
        context = {}
        
        try:
            db = await get_database()
            
            # Get customer info
            if entities.get("customer_id"):
                customer_query = "SELECT * FROM customers WHERE customer_id = :customer_id"
                customer = await db.fetch_one(customer_query, {"customer_id": entities["customer_id"]})
                if customer:
                    context["customer"] = dict(customer)
            
            # Get order info
            if entities.get("order_id"):
                order_query = """
                    SELECT o.*, c.name as customer_name, p.name as product_name 
                    FROM orders o 
                    LEFT JOIN customers c ON o.customer_id = c.customer_id 
                    LEFT JOIN products p ON o.product_id = p.product_id 
                    WHERE o.order_id = :order_id
                """
                order = await db.fetch_one(order_query, {"order_id": entities["order_id"]})
                if order:
                    context["order"] = dict(order)
            
            # Get product info
            if entities.get("product_name"):
                product_query = "SELECT * FROM products WHERE name ILIKE :product_name LIMIT 1"
                product = await db.fetch_one(product_query, {"product_name": f"%{entities['product_name']}%"})
                if product:
                    context["product"] = dict(product)
            
        except Exception as e:
            logger.error(f"Database context fetch failed: {e}")
        
        return context
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            
            # Quick test request
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            
            response_time = time.time() - start_time
            
            return {
                "status": "healthy",
                "model": self.model,
                "response_time": response_time,
                "initialized": self.is_initialized
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized
            }


# Global LLM service instance
llm_service = OpenAILLMService()


async def generate_response(
    user_message: str,
    context: Optional[Dict[str, Any]] = None,
    conversation_history: Optional[List[Dict[str, str]]] = None
) -> Dict[str, Any]:
    """Convenience function for generating responses."""
    return await llm_service.generate_response(user_message, context, conversation_history)


async def analyze_and_respond(user_message: str) -> Dict[str, Any]:
    """
    Analyze user intent and generate contextual response.
    
    Args:
        user_message: User's input message
        
    Returns:
        Complete response with intent analysis and context
    """
    try:
        # Analyze intent
        intent_result = await llm_service.analyze_intent(user_message)
        
        # Get context from database
        context = {}
        if intent_result["success"] and intent_result["entities"]:
            context = await llm_service.get_context_from_database(intent_result["entities"])
        
        # Generate response with context
        response_result = await llm_service.generate_response(user_message, context)
        
        return {
            "response": response_result["response"],
            "intent": intent_result.get("intent", "other"),
            "entities": intent_result.get("entities", {}),
            "context": context,
            "processing_time": response_result.get("processing_time", 0),
            "success": response_result["success"]
        }
        
    except Exception as e:
        logger.error(f"Analyze and respond failed: {e}")
        return {
            "response": "I apologize, but I'm experiencing technical difficulties. Please try again.",
            "error": str(e),
            "success": False
        }


async def initialize_llm() -> bool:
    """Initialize LLM service."""
    try:
        await llm_service.initialize()
        return True
    except Exception as e:
        logger.error(f"LLM initialization failed: {e}")
        return False
