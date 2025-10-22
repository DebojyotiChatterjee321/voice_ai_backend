from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel, EmailStr, Field, validator


class CustomerBase(BaseModel):
    """Base customer schema with common fields."""
    name: str = Field(..., min_length=1, max_length=255, description="Customer full name")
    email: EmailStr = Field(..., description="Customer email address")
    phone: Optional[str] = Field(None, max_length=20, description="Customer phone number")
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is not None:
            # Remove common phone formatting characters
            cleaned = ''.join(c for c in v if c.isdigit() or c in ['+', '-', ' ', '(', ')'])
            if len(cleaned.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) < 10:
                raise ValueError('Phone number must have at least 10 digits')
        return v


class CustomerCreate(CustomerBase):
    """Schema for creating a new customer."""
    customer_id: str = Field(..., min_length=1, max_length=50, description="Unique customer identifier")
    
    @validator('customer_id')
    def validate_customer_id(cls, v):
        if not v.startswith('CUST'):
            raise ValueError('Customer ID must start with "CUST"')
        return v


class CustomerUpdate(BaseModel):
    """Schema for updating customer information."""
    name: Optional[str] = Field(None, min_length=1, max_length=255)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    
    @validator('phone')
    def validate_phone(cls, v):
        if v is not None:
            cleaned = ''.join(c for c in v if c.isdigit() or c in ['+', '-', ' ', '(', ')'])
            if len(cleaned.replace('+', '').replace('-', '').replace(' ', '').replace('(', '').replace(')', '')) < 10:
                raise ValueError('Phone number must have at least 10 digits')
        return v


class CustomerResponse(CustomerBase):
    """Schema for customer API responses."""
    customer_id: str
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class CustomerWithOrders(CustomerResponse):
    """Schema for customer with their orders."""
    orders: List['OrderResponse'] = []
    
    class Config:
        from_attributes = True


class CustomerSearch(BaseModel):
    """Schema for customer search parameters."""
    name: Optional[str] = Field(None, description="Search by customer name (partial match)")
    email: Optional[str] = Field(None, description="Search by email (partial match)")
    phone: Optional[str] = Field(None, description="Search by phone number")
    customer_id: Optional[str] = Field(None, description="Search by customer ID")


# Forward reference resolution
from .order import OrderResponse
CustomerWithOrders.model_rebuild()
