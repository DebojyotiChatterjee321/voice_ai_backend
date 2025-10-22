from sqlalchemy import Column, String, Integer, Numeric, DateTime, Text, ForeignKey, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from enum import Enum as PyEnum
from app.db.connection import Base


class OrderStatus(PyEnum):
    """Order status enumeration."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SHIPPED = "shipped"
    OUT_FOR_DELIVERY = "out_for_delivery"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    RETURNED = "returned"


class Order(Base):
    """Order model for e-commerce transactions."""
    
    __tablename__ = "orders"
    
    # Primary key - using string ID as in sample data
    order_id = Column(String(50), primary_key=True, index=True)
    
    # Foreign keys
    customer_id = Column(String(50), ForeignKey("customers.customer_id"), nullable=False, index=True)
    product_id = Column(String(50), ForeignKey("products.product_id"), nullable=False, index=True)
    
    # Order details
    quantity = Column(Integer, nullable=False, default=1)
    total_amount = Column(Numeric(10, 2), nullable=False)
    status = Column(Enum(OrderStatus), nullable=False, default=OrderStatus.PENDING, index=True)
    
    # Shipping information
    shipping_address = Column(Text, nullable=False)
    estimated_delivery = Column(DateTime(timezone=True), nullable=True)
    actual_delivery = Column(DateTime(timezone=True), nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    customer = relationship("Customer", back_populates="orders")
    product = relationship("Product", back_populates="orders")
    
    def __repr__(self):
        return f"<Order(order_id='{self.order_id}', customer_id='{self.customer_id}', status='{self.status.value}')>"
    
    @property
    def is_delivered(self):
        """Check if order is delivered."""
        return self.status == OrderStatus.DELIVERED
    
    @property
    def is_active(self):
        """Check if order is active (not cancelled or returned)."""
        return self.status not in [OrderStatus.CANCELLED, OrderStatus.RETURNED]
    
    @property
    def can_be_cancelled(self):
        """Check if order can be cancelled."""
        return self.status in [OrderStatus.PENDING, OrderStatus.CONFIRMED]
    
    @property
    def unit_price(self):
        """Calculate unit price."""
        if self.quantity > 0:
            return float(self.total_amount) / self.quantity
        return 0.0
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "product_id": self.product_id,
            "quantity": self.quantity,
            "total_amount": float(self.total_amount) if self.total_amount else None,
            "unit_price": self.unit_price,
            "status": self.status.value if self.status else None,
            "shipping_address": self.shipping_address,
            "estimated_delivery": self.estimated_delivery.isoformat() if self.estimated_delivery else None,
            "actual_delivery": self.actual_delivery.isoformat() if self.actual_delivery else None,
            "is_delivered": self.is_delivered,
            "is_active": self.is_active,
            "can_be_cancelled": self.can_be_cancelled,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
