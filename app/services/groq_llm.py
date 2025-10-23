"""
Groq LLM service using Llama 3.3 70B for ultra-fast inference.
Achieves 800+ tokens/sec for sub-500ms response times.
"""

import asyncio
import logging
import time
from typing import Optional, Dict, Any, List, AsyncGenerator
import json

from groq import AsyncGroq
import httpx

from app.config import settings

logger = logging.getLogger(__name__)


class GroqLLMService:
    """Asynchronous Groq LLM service optimized for ultra-low latency."""
    
    def __init__(self):
        self.client = None
        self.model = settings.groq_llm_model
        self.max_tokens = settings.openai_max_tokens
        self.is_initialized = False
        
        # Performance settings
        self.temperature = 0.1
        self.timeout = 10.0
        
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
        """Initialize Groq client."""
        if self.is_initialized:
            return
        
        try:
            logger.info("Initializing Groq LLM service")
            
            # Initialize async client
            self.client = AsyncGroq(
                api_key=settings.groq_api_key,
                timeout=httpx.Timeout(self.timeout),
                max_retries=2
            )
            
            # Test connection
            await self._test_connection()
            
            self.is_initialized = True
            logger.info(f"✅ Groq LLM service initialized successfully (Model: {self.model})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM service: {e}")
            raise
    
    async def _test_connection(self):
        """Test Groq API connection."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
                temperature=0
            )
            logger.info("Groq API connection test successful")
        except Exception as e:
            logger.error(f"Groq API connection test failed: {e}")
            raise
    
    async def generate_response(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Generate response to user message using Groq.
        
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
            
            # Generate response with streaming for ultra-low latency
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stream=True  # Critical for low latency
            )
            
            # Collect streaming response
            assistant_message = ""
            first_token_time = None
            
            async for chunk in response:
                if chunk.choices[0].delta.content is not None:
                    if first_token_time is None:
                        first_token_time = time.time() - start_time
                    assistant_message += chunk.choices[0].delta.content
            
            processing_time = time.time() - start_time
            
            logger.info(f"🚀 Groq LLM: {processing_time*1000:.0f}ms (TTFT: {(first_token_time or 0)*1000:.0f}ms)")
            
            return {
                "response": assistant_message.strip(),
                "processing_time": processing_time,
                "time_to_first_token": first_token_time,
                "model": self.model,
                "tokens_used": len(assistant_message.split()),
                "success": True,
                "service": "groq",
                "cached": False
            }
            
        except Exception as e:
            logger.error(f"Groq LLM response generation failed: {e}")
            return {
                "response": "I apologize, but I'm experiencing technical difficulties. Please try again or contact our support team.",
                "error": str(e),
                "processing_time": time.time() - start_time,
                "success": False,
                "service": "groq"
            }
    
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
            logger.error(f"Groq streaming response failed: {e}")
            yield "I apologize, but I'm experiencing technical difficulties."
    
    async def _build_messages(
        self,
        user_message: str,
        context: Optional[Dict[str, Any]],
        conversation_history: Optional[List[Dict[str, str]]]
    ) -> List[Dict[str, str]]:
        """Build message array for Groq API."""
        messages = [{"role": "system", "content": self.system_prompt}]
        
        # Add context if provided
        if context:
            context_message = await self._format_context(context)
            if context_message:
                messages.append({"role": "system", "content": context_message})
        
        # Add conversation history (keep it short for speed)
        if conversation_history:
            recent_history = conversation_history[-4:]
            messages.extend(recent_history)
        
        # Add current user message
        messages.append({"role": "user", "content": user_message})
        
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
    
    async def health_check(self) -> Dict[str, Any]:
        """Check service health."""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            start_time = time.time()
            
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
                "initialized": self.is_initialized,
                "service": "groq"
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "initialized": self.is_initialized,
                "service": "groq"
            }


# Global Groq LLM service instance
groq_llm_service = GroqLLMService()


async def test_groq_llm():
    """Test Groq LLM service."""
    print("\n🧪 Testing Groq LLM Service...")
    
    try:
        await groq_llm_service.initialize()
        print("✅ Groq LLM initialized")
        
        # Test response
        start = time.time()
        result = await groq_llm_service.generate_response("What is your return policy?")
        latency = time.time() - start
        
        print(f"✅ Response generated in {latency*1000:.0f}ms")
        print(f"   TTFT: {result.get('time_to_first_token', 0)*1000:.0f}ms")
        print(f"   Response: {result['response'][:100]}...")
        
        if latency < 0.5:
            print("🚀 Excellent! Sub-500ms latency achieved!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_groq_llm())
