from sqlalchemy import Column, String, Text, Numeric, Integer, DateTime, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.connection import Base


class Product(Base):
    """Product model for e-commerce catalog."""
    
    __tablename__ = "products"
    
    # Primary key - using string ID as in sample data
    product_id = Column(String(50), primary_key=True, index=True)
    
    # Product information
    name = Column(String(500), nullable=False, index=True)
    description = Column(Text, nullable=True)
    price = Column(Numeric(10, 2), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    
    # Product details for customer support
    use_case = Column(Text, nullable=True)
    application = Column(Text, nullable=True)
    
    # Inventory
    stock_quantity = Column(Integer, nullable=False, default=0, index=True)
    
    # Product metadata (JSON field for flexible attributes)
    product_metadata = Column(JSON, nullable=True)
    
    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    # Relationships
    orders = relationship("Order", back_populates="product")
    
    def __repr__(self):
        return f"<Product(product_id='{self.product_id}', name='{self.name}', price={self.price})>"
    
    @property
    def is_in_stock(self):
        """Check if product is in stock."""
        return self.stock_quantity > 0
    
    @property
    def brand(self):
        """Get brand from metadata."""
        if self.product_metadata and isinstance(self.product_metadata, dict):
            return self.product_metadata.get('brand')
        return None
    
    def to_dict(self):
        """Convert model to dictionary."""
        return {
            "product_id": self.product_id,
            "name": self.name,
            "description": self.description,
            "price": float(self.price) if self.price else None,
            "category": self.category,
            "use_case": self.use_case,
            "application": self.application,
            "stock_quantity": self.stock_quantity,
            "product_metadata": self.product_metadata,
            "is_in_stock": self.is_in_stock,
            "brand": self.brand,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
