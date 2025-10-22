"""
Database initialization and testing utilities.
"""

import asyncio
import logging
from typing import Dict, Any

from sqlalchemy import text
from app.db.connection import (
    connect_database,
    disconnect_database,
    test_database_connection,
    get_database,
    get_db_connection,
    db_manager,
    Base,
    async_engine,
)
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def create_database_if_not_exists() -> bool:
    """Create database if it doesn't exist."""
    try:
        # Connect to postgres database to create our target database
        postgres_url = settings.database_url.replace(f"/{settings.database_name}", "/postgres")
        
        import asyncpg
        conn = await asyncpg.connect(postgres_url)
        
        # Check if database exists
        exists = await conn.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", 
            settings.database_name
        )
        
        if not exists:
            # Create database
            await conn.execute(f'CREATE DATABASE "{settings.database_name}"')
            logger.info(f"Database '{settings.database_name}' created successfully")
            await conn.close()
            return True
        else:
            logger.info(f"Database '{settings.database_name}' already exists")
            await conn.close()
            return False
            
    except Exception as e:
        logger.error(f"Error creating database: {e}")
        return False


async def create_tables() -> bool:
    """Create all tables defined in models."""
    try:
        async with async_engine.begin() as conn:
            # Create all tables
            await conn.run_sync(Base.metadata.create_all)
            logger.info("All tables created successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
        return False


async def drop_tables() -> bool:
    """Drop all tables (use with caution!)."""
    try:
        async with async_engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
            logger.info("All tables dropped successfully")
            return True
            
    except Exception as e:
        logger.error(f"Error dropping tables: {e}")
        return False


async def test_database_operations() -> Dict[str, Any]:
    """Test basic database operations."""
    results = {
        "connection_test": False,
        "query_test": False,
        "transaction_test": False,
        "pool_test": False,
        "errors": []
    }
    
    try:
        # Test 1: Basic connection
        logger.info("Testing database connection...")
        connection_result = await test_database_connection()
        results["connection_test"] = connection_result.get("health_check", False)
        
        if not results["connection_test"]:
            results["errors"].append("Connection test failed")
            return results
        
        # Test 2: Basic query
        logger.info("Testing basic query...")
        db = await get_database()
        query_result = await db.fetch_one("SELECT NOW() as current_time, 'Hello Database!' as message")
        results["query_test"] = query_result is not None
        
        if results["query_test"]:
            logger.info(f"Query result: {dict(query_result)}")
        else:
            results["errors"].append("Basic query test failed")
        
        # Test 3: Transaction test
        logger.info("Testing transaction...")
        db = await get_database()
        try:
            # Start a transaction and test it
            query = """
            DO $$
            BEGIN
                -- This is a transaction test
                IF (SELECT 1 + 1) = 2 THEN
                    -- All good
                    NULL;
                ELSE
                    RAISE EXCEPTION 'Transaction test failed';
                END IF;
            END $$;
            """
            await db.execute(query)
            results["transaction_test"] = True
        except Exception as e:
            logger.error(f"Transaction test failed: {e}")
            results["transaction_test"] = False
        
        if not results["transaction_test"]:
            results["errors"].append("Transaction test failed")
        
        # Test 4: Connection pool test
        logger.info("Testing connection pool...")
        pool_info = {
            "size": db_manager._connection_pool.get_size(),
            "idle": db_manager._connection_pool.get_idle_size(),
            "min_size": settings.database_min_connections,
            "max_size": settings.database_max_connections,
        }
        results["pool_test"] = pool_info["size"] >= pool_info["min_size"]
        results["pool_info"] = pool_info
        
        if not results["pool_test"]:
            results["errors"].append("Connection pool test failed")
        
        logger.info("All database tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Database test error: {e}")
        results["errors"].append(str(e))
    
    return results


async def initialize_database() -> bool:
    """Initialize database with tables and test data."""
    try:
        logger.info("Starting database initialization...")
        
        # Step 1: Create database if needed
        await create_database_if_not_exists()
        
        # Step 2: Connect to database
        await connect_database()
        
        # Step 3: Create tables
        tables_created = await create_tables()
        
        # Step 4: Test operations
        test_results = await test_database_operations()
        
        success = (
            tables_created and 
            test_results["connection_test"] and 
            test_results["query_test"] and 
            test_results["transaction_test"]
        )
        
        if success:
            logger.info("Database initialization completed successfully!")
        else:
            logger.error(f"Database initialization failed. Errors: {test_results.get('errors', [])}")
        
        return success
        
    except Exception as e:
        logger.error(f"Database initialization error: {e}")
        return False


async def reset_database() -> bool:
    """Reset database by dropping and recreating all tables."""
    try:
        logger.warning("Resetting database - this will delete all data!")
        
        await connect_database()
        
        # Drop all tables
        dropped = await drop_tables()
        if not dropped:
            return False
        
        # Recreate tables
        created = await create_tables()
        if not created:
            return False
        
        logger.info("Database reset completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Database reset error: {e}")
        return False


async def main():
    """Main function for running database operations."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.db.init_db [init|test|reset]")
        return
    
    command = sys.argv[1].lower()
    
    try:
        if command == "init":
            success = await initialize_database()
            print(f"Database initialization: {'SUCCESS' if success else 'FAILED'}")
            
        elif command == "test":
            await connect_database()
            results = await test_database_operations()
            print("Database Test Results:")
            for key, value in results.items():
                print(f"  {key}: {value}")
                
        elif command == "reset":
            success = await reset_database()
            print(f"Database reset: {'SUCCESS' if success else 'FAILED'}")
            
        else:
            print(f"Unknown command: {command}")
            print("Available commands: init, test, reset")
            
    except Exception as e:
        logger.error(f"Command execution error: {e}")
        print(f"Error: {e}")
        
    finally:
        await disconnect_database()


if __name__ == "__main__":
    asyncio.run(main())
