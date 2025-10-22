from typing import Optional, Dict, Any, List
from datetime import datetime
from decimal import Decimal
from pydantic import BaseModel, Field, validator


class ProductBase(BaseModel):
    """Base product schema with common fields."""
    name: str = Field(..., min_length=1, max_length=500, description="Product name")
    description: Optional[str] = Field(None, description="Product description")
    price: Decimal = Field(..., gt=0, decimal_places=2, description="Product price")
    category: str = Field(..., min_length=1, max_length=100, description="Product category")
    use_case: Optional[str] = Field(None, description="Product use cases")
    application: Optional[str] = Field(None, description="Product applications")
    stock_quantity: int = Field(..., ge=0, description="Stock quantity")
    product_metadata: Optional[Dict[str, Any]] = Field(None, description="Additional product metadata")


class ProductCreate(ProductBase):
    """Schema for creating a new product."""
    product_id: str = Field(..., min_length=1, max_length=50, description="Unique product identifier")
    
    @validator('product_id')
    def validate_product_id(cls, v):
        if not v.startswith('PROD'):
            raise ValueError('Product ID must start with "PROD"')
        return v


class ProductUpdate(BaseModel):
    """Schema for updating product information."""
    name: Optional[str] = Field(None, min_length=1, max_length=500)
    description: Optional[str] = None
    price: Optional[Decimal] = Field(None, gt=0, decimal_places=2)
    category: Optional[str] = Field(None, min_length=1, max_length=100)
    use_case: Optional[str] = None
    application: Optional[str] = None
    stock_quantity: Optional[int] = Field(None, ge=0)
    product_metadata: Optional[Dict[str, Any]] = None


class ProductResponse(ProductBase):
    """Schema for product API responses."""
    product_id: str
    is_in_stock: bool
    brand: Optional[str]
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class ProductWithOrders(ProductResponse):
    """Schema for product with order history."""
    orders: List['OrderResponse'] = []
    
    class Config:
        from_attributes = True


class ProductSearch(BaseModel):
    """Schema for product search parameters."""
    name: Optional[str] = Field(None, description="Search by product name (partial match)")
    category: Optional[str] = Field(None, description="Filter by category")
    brand: Optional[str] = Field(None, description="Filter by brand")
    min_price: Optional[Decimal] = Field(None, ge=0, description="Minimum price filter")
    max_price: Optional[Decimal] = Field(None, ge=0, description="Maximum price filter")
    in_stock_only: Optional[bool] = Field(False, description="Show only products in stock")
    product_id: Optional[str] = Field(None, description="Search by product ID")
    
    @validator('max_price')
    def validate_price_range(cls, v, values):
        if v is not None and 'min_price' in values and values['min_price'] is not None:
            if v < values['min_price']:
                raise ValueError('max_price must be greater than or equal to min_price')
        return v


class ProductInventoryUpdate(BaseModel):
    """Schema for updating product inventory."""
    stock_quantity: int = Field(..., ge=0, description="New stock quantity")
    operation: Optional[str] = Field("set", description="Operation type: 'set', 'add', 'subtract'")
    
    @validator('operation')
    def validate_operation(cls, v):
        if v not in ['set', 'add', 'subtract']:
            raise ValueError('Operation must be one of: set, add, subtract')
        return v


class ProductStats(BaseModel):
    """Schema for product statistics."""
    total_products: int
    products_in_stock: int
    products_out_of_stock: int
    categories: List[str]
    average_price: Decimal
    total_inventory_value: Decimal


# Forward reference resolution
from .order import OrderResponse
ProductWithOrders.model_rebuild()
