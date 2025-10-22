from typing import Optional, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator
from enum import Enum

from app.models.order import OrderStatus


class OrderStatusEnum(str, Enum):
    """Order status enumeration for API."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class OrderBase(BaseModel):
    """Base order schema with common fields."""
    customer_id: str = Field(..., min_length=1, max_length=50, description="Customer identifier")
    product_id: str = Field(..., min_length=1, max_length=50, description="Product identifier")
    quantity: int = Field(..., gt=0, description="Order quantity")
    total_amount: Decimal = Field(..., gt=0, decimal_places=2, description="Total order amount")
    shipping_address: str = Field(..., min_length=1, description="Shipping address")
    estimated_delivery: Optional[datetime] = Field(None, description="Estimated delivery date")
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v.startswith('CUST'):
            raise ValueError('Customer ID must start with "CUST"')
        return v
    
    @validator('product_id')
    def validate_product_id(cls, v):
        if not v.startswith('PROD'):
            raise ValueError('Product ID must start with "PROD"')
        return v


class OrderCreate(OrderBase):
    """Schema for creating a new order."""
    order_id: str = Field(..., min_length=1, max_length=50, description="Unique order identifier")
    status: Optional[OrderStatusEnum] = Field(OrderStatusEnum.PENDING, description="Order status")
    
    @validator('order_id')
    def validate_order_id(cls, v):
        if not v.startswith('ORD'):
            raise ValueError('Order ID must start with "ORD"')
        return v


class OrderUpdate(BaseModel):
    """Schema for updating order information."""
    quantity: Optional[int] = Field(None, gt=0)
    total_amount: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    status: Optional[OrderStatusEnum] = None
    shipping_address: Optional[str] = Field(None, min_length=1)
    estimated_delivery: Optional[datetime] = None
    actual_delivery: Optional[datetime] = None


class OrderStatusUpdate(BaseModel):
    """Schema for updating order status."""
    status: OrderStatusEnum = Field(..., description="New order status")
    actual_delivery: Optional[datetime] = Field(None, description="Actual delivery date (for delivered status)")
    
    @validator('actual_delivery')
    def validate_delivery_date(cls, v, values):
        if 'status' in values and values['status'] == OrderStatusEnum.DELIVERED and v is None:
            raise ValueError('actual_delivery is required when status is delivered')
        return v


class OrderResponse(OrderBase):
    """Schema for order API responses."""
    order_id: str
    status: OrderStatusEnum
    actual_delivery: Optional[datetime]
    unit_price: Decimal
    is_delivered: bool
    is_active: bool
    can_be_cancelled: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class OrderWithDetails(OrderResponse):
    """Schema for order with customer and product details."""
    customer: Optional['CustomerResponse'] = None
    product: Optional['ProductResponse'] = None
    
    class Config:
        from_attributes = True


class OrderSearch(BaseModel):
    """Schema for order search parameters."""
    order_id: Optional[str] = Field(None, description="Search by order ID")
    customer_id: Optional[str] = Field(None, description="Filter by customer ID")
    product_id: Optional[str] = Field(None, description="Filter by product ID")
    status: Optional[OrderStatusEnum] = Field(None, description="Filter by order status")
    date_from: Optional[datetime] = Field(None, description="Filter orders from this date")
    date_to: Optional[datetime] = Field(None, description="Filter orders to this date")
    min_amount: Optional[Decimal] = Field(None, ge=0, description="Minimum order amount")
    max_amount: Optional[Decimal] = Field(None, ge=0, description="Maximum order amount")
    
    @validator('date_to')
    def validate_date_range(cls, v, values):
        if v is not None and 'date_from' in values and values['date_from'] is not None:
            if v < values['date_from']:
                raise ValueError('date_to must be greater than or equal to date_from')
        return v
    
    @validator('max_amount')
    def validate_amount_range(cls, v, values):
        if v is not None and 'min_amount' in values and values['min_amount'] is not None:
            if v < values['min_amount']:
                raise ValueError('max_amount must be greater than or equal to min_amount')
        return v


class OrderStats(BaseModel):
    """Schema for order statistics."""
    total_orders: int
    orders_by_status: dict[OrderStatusEnum, int]
    total_revenue: Decimal
    average_order_value: Decimal
    orders_today: int
    orders_this_week: int
    orders_this_month: int


class OrderDeliveryUpdate(BaseModel):
    """Schema for delivery updates."""
    estimated_delivery: Optional[datetime] = Field(None, description="New estimated delivery date")
    tracking_info: Optional[str] = Field(None, description="Tracking information")
    delivery_notes: Optional[str] = Field(None, description="Delivery notes")


# Forward reference resolution
from .customer import CustomerResponse
from .product import ProductResponse
OrderWithDetails.model_rebuild()
