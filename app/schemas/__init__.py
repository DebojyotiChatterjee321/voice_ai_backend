"""
Pydantic schemas for API request/response validation.
"""

from .customer import (
    CustomerBase,
    CustomerCreate,
    CustomerUpdate,
    CustomerResponse,
    CustomerWithOrders,
    CustomerSearch
)

from .product import (
    ProductBase,
    ProductCreate,
    ProductUpdate,
    ProductResponse,
    ProductWithOrders,
    ProductSearch,
    ProductInventoryUpdate,
    ProductStats
)

from .order import (
    OrderBase,
    OrderCreate,
    OrderUpdate,
    OrderStatusUpdate,
    OrderResponse,
    OrderWithDetails,
    OrderSearch,
    OrderStats,
    OrderDeliveryUpdate,
    OrderStatusEnum
)

# Export all schemas
__all__ = [
    # Customer schemas
    "CustomerBase",
    "CustomerCreate", 
    "CustomerUpdate",
    "CustomerResponse",
    "CustomerWithOrders",
    "CustomerSearch",
    
    # Product schemas
    "ProductBase",
    "ProductCreate",
    "ProductUpdate", 
    "ProductResponse",
    "ProductWithOrders",
    "ProductSearch",
    "ProductInventoryUpdate",
    "ProductStats",
    
    # Order schemas
    "OrderBase",
    "OrderCreate",
    "OrderUpdate",
    "OrderStatusUpdate",
    "OrderResponse", 
    "OrderWithDetails",
    "OrderSearch",
    "OrderStats",
    "OrderDeliveryUpdate",
    "OrderStatusEnum"
]