"""
Data seeding script for the e-commerce customer support database.
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.db.connection import connect_database, disconnect_database, AsyncSessionLocal
from app.models import Customer, Product, Order, OrderStatus
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSeeder:
    """Data seeding utility class."""
    
    def __init__(self):
        self.data_dir = Path(__file__).parent.parent.parent / "data"
        self.session: AsyncSession = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        await connect_database()
        self.session = AsyncSessionLocal()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
        await disconnect_database()
    
    def load_json_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from JSON file."""
        file_path = self.data_dir / filename
        if not file_path.exists():
            logger.error(f"Data file not found: {file_path}")
            return []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded {len(data)} records from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return []
    
    async def clear_all_data(self) -> bool:
        """Clear all existing data from tables."""
        try:
            logger.info("Clearing existing data...")
            
            # Delete in order to respect foreign key constraints
            await self.session.execute(delete(Order))
            await self.session.execute(delete(Product))
            await self.session.execute(delete(Customer))
            
            await self.session.commit()
            logger.info("All existing data cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing data: {e}")
            await self.session.rollback()
            return False
    
    async def seed_customers(self) -> bool:
        """Seed customer data."""
        try:
            logger.info("Seeding customers...")
            customers_data = self.load_json_data("customers.json")
            
            if not customers_data:
                logger.warning("No customer data to seed")
                return True
            
            customers = []
            for data in customers_data:
                customer = Customer(
                    customer_id=data["customer_id"],
                    name=data["name"],
                    email=data["email"],
                    phone=data.get("phone")
                )
                customers.append(customer)
            
            self.session.add_all(customers)
            await self.session.commit()
            
            logger.info(f"Successfully seeded {len(customers)} customers")
            return True
            
        except Exception as e:
            logger.error(f"Error seeding customers: {e}")
            await self.session.rollback()
            return False
    
    async def seed_products(self) -> bool:
        """Seed product data."""
        try:
            logger.info("Seeding products...")
            products_data = self.load_json_data("products.json")
            
            if not products_data:
                logger.warning("No product data to seed")
                return True
            
            products = []
            for data in products_data:
                product = Product(
                    product_id=data["product_id"],
                    name=data["name"],
                    description=data.get("description"),
                    price=data["price"],
                    category=data["category"],
                    use_case=data.get("use_case"),
                    application=data.get("application"),
                    stock_quantity=data.get("stock_quantity", 0),
                    product_metadata=data.get("product_metadata")
                )
                products.append(product)
            
            self.session.add_all(products)
            await self.session.commit()
            
            logger.info(f"Successfully seeded {len(products)} products")
            return True
            
        except Exception as e:
            logger.error(f"Error seeding products: {e}")
            await self.session.rollback()
            return False
    
    async def seed_orders(self) -> bool:
        """Seed order data."""
        try:
            logger.info("Seeding orders...")
            orders_data = self.load_json_data("orders.json")
            
            if not orders_data:
                logger.warning("No order data to seed")
                return True
            
            orders = []
            for data in orders_data:
                # Parse estimated delivery date
                estimated_delivery = None
                if data.get("estimated_delivery"):
                    try:
                        estimated_delivery = datetime.fromisoformat(data["estimated_delivery"])
                    except ValueError:
                        logger.warning(f"Invalid date format for order {data['order_id']}: {data['estimated_delivery']}")
                
                # Map status string to enum
                status_map = {
                    "pending": OrderStatus.PENDING,
                    "confirmed": OrderStatus.CONFIRMED,
                    "shipped": OrderStatus.SHIPPED,
                    "out_for_delivery": OrderStatus.OUT_FOR_DELIVERY,
                    "delivered": OrderStatus.DELIVERED,
                    "cancelled": OrderStatus.CANCELLED,
                    "returned": OrderStatus.RETURNED
                }
                
                status = status_map.get(data.get("status", "pending"), OrderStatus.PENDING)
                
                order = Order(
                    order_id=data["order_id"],
                    customer_id=data["customer_id"],
                    product_id=data["product_id"],
                    quantity=data["quantity"],
                    total_amount=data["total_amount"],
                    status=status,
                    shipping_address=data["shipping_address"],
                    estimated_delivery=estimated_delivery
                )
                orders.append(order)
            
            self.session.add_all(orders)
            await self.session.commit()
            
            logger.info(f"Successfully seeded {len(orders)} orders")
            return True
            
        except Exception as e:
            logger.error(f"Error seeding orders: {e}")
            await self.session.rollback()
            return False
    
    async def verify_data(self) -> Dict[str, int]:
        """Verify seeded data by counting records."""
        try:
            customer_count = await self.session.scalar(select(Customer).count())
            product_count = await self.session.scalar(select(Product).count())
            order_count = await self.session.scalar(select(Order).count())
            
            counts = {
                "customers": customer_count or 0,
                "products": product_count or 0,
                "orders": order_count or 0
            }
            
            logger.info(f"Data verification: {counts}")
            return counts
            
        except Exception as e:
            logger.error(f"Error verifying data: {e}")
            return {"customers": 0, "products": 0, "orders": 0}
    
    async def seed_all_data(self, clear_existing: bool = True) -> bool:
        """Seed all data from JSON files."""
        try:
            logger.info("Starting data seeding process...")
            
            if clear_existing:
                success = await self.clear_all_data()
                if not success:
                    return False
            
            # Seed in order to respect foreign key constraints
            success = await self.seed_customers()
            if not success:
                return False
            
            success = await self.seed_products()
            if not success:
                return False
            
            success = await self.seed_orders()
            if not success:
                return False
            
            # Verify the seeded data
            counts = await self.verify_data()
            
            logger.info("Data seeding completed successfully!")
            logger.info(f"Final counts: {counts}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in seed_all_data: {e}")
            return False


async def seed_database(clear_existing: bool = True) -> bool:
    """Main function to seed the database."""
    async with DataSeeder() as seeder:
        return await seeder.seed_all_data(clear_existing)


async def verify_database() -> Dict[str, int]:
    """Verify database contents."""
    async with DataSeeder() as seeder:
        return await seeder.verify_data()


async def clear_database() -> bool:
    """Clear all data from database."""
    async with DataSeeder() as seeder:
        return await seeder.clear_all_data()


async def main():
    """Main CLI function."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.db.seed_data [seed|verify|clear]")
        print("Commands:")
        print("  seed   - Seed database with sample data (clears existing data)")
        print("  verify - Verify database contents")
        print("  clear  - Clear all data from database")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "seed":
            success = await seed_database(clear_existing=True)
            if success:
                print("✅ Database seeded successfully!")
            else:
                print("❌ Database seeding failed!")
                sys.exit(1)
                
        elif command == "verify":
            counts = await verify_database()
            print("📊 Database Contents:")
            for table, count in counts.items():
                print(f"  {table}: {count} records")
                
        elif command == "clear":
            success = await clear_database()
            if success:
                print("✅ Database cleared successfully!")
            else:
                print("❌ Database clearing failed!")
                sys.exit(1)
                
        else:
            print(f"Unknown command: {command}")
            print("Available commands: seed, verify, clear")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
