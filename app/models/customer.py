from sqlalchemy import Column, String, DateTime, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.connection import Base


class Customer(Base):
    """Customer model for e-commerce customer support."""
    
    __tablename__ = "customers"
    
    # Primary key - using string ID as in sample data
    customer_id = Column(String(50), primary_key=True, index=True)
    
    # Customer information
    name = Column(String(255), nullable=False, index=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    phone = Column(String(20), nullable=True, index=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    orders = relationship("Order", back_populates="customer", cascade="all, delete-orphan")
    
    def __repr__(self):
        return f"<Customer(customer_id='{self.customer_id}', name='{self.name}', email='{self.email}')>"
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "customer_id": self.customer_id,
            "name": self.name,
            "email": self.email,
            "phone": self.phone,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
