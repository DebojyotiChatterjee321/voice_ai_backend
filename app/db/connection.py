import asyncio
import logging
from typing import Optional
from contextlib import asynccontextmanager

import asyncpg
from databases import Database
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool

from app.config import settings

# Configure logging
logger = logging.getLogger(__name__)

# SQLAlchemy setup
Base = declarative_base()
metadata = MetaData()

# Global database instance
database: Optional[Database] = None

# Async engine for SQLAlchemy ORM
async_engine = create_async_engine(
    settings.async_database_url,
    echo=settings.debug,
    poolclass=NullPool,  # Use databases package for connection pooling
    future=True,
)

# Session factory for SQLAlchemy ORM
AsyncSessionLocal = async_sessionmaker(
    bind=async_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
    autocommit=False,
)

# Sync engine for migrations
sync_engine = create_engine(
    settings.sync_database_url,
    echo=settings.debug,
    pool_pre_ping=True,
    pool_recycle=settings.database_pool_recycle,
)


class DatabaseManager:
    """Database manager for handling connections and lifecycle."""
    
    def __init__(self):
        self.database: Optional[Database] = None
        self._connection_pool: Optional[asyncpg.Pool] = None
    
    async def connect(self) -> None:
        """Initialize database connections with optimized settings."""
        try:
            # Create asyncpg connection pool for raw queries
            self._connection_pool = await asyncpg.create_pool(
                settings.database_url,
                min_size=settings.database_min_connections,
                max_size=settings.database_max_connections,
                max_queries=50000,  # High query limit for performance
                max_inactive_connection_lifetime=300,  # 5 minutes
                timeout=settings.database_pool_timeout,
                command_timeout=60,  # Command timeout
                server_settings={
                    'application_name': settings.app_name,
                    'tcp_keepalives_idle': '600',
                    'tcp_keepalives_interval': '30',
                    'tcp_keepalives_count': '3',
                }
            )
            
            # Create databases instance with the pool
            self.database = Database(
                settings.database_url,
                min_size=settings.database_min_connections,
                max_size=settings.database_max_connections,
                ssl=False,  # Set to True in production with SSL
            )
            
            await self.database.connect()
            
            logger.info(
                f"Database connected successfully. "
                f"Pool: {settings.database_min_connections}-{settings.database_max_connections} connections"
            )
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connections."""
        try:
            if self.database:
                await self.database.disconnect()
                logger.info("Database disconnected")
            
            if self._connection_pool:
                await self._connection_pool.close()
                logger.info("Connection pool closed")
                
        except Exception as e:
            logger.error(f"Error disconnecting from database: {e}")
    
    async def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            if not self.database:
                return False
            
            query = "SELECT 1 as health_check"
            result = await self.database.fetch_one(query)
            return result is not None and result["health_check"] == 1
            
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    async def get_connection(self):
        """Get a database connection from the pool."""
        if not self._connection_pool:
            raise RuntimeError("Database not connected")
        return self._connection_pool.acquire()
    
    @property
    def is_connected(self) -> bool:
        """Check if database is connected."""
        return self.database is not None and self._connection_pool is not None


# Global database manager instance
db_manager = DatabaseManager()


async def get_database() -> Database:
    """Get the database instance."""
    if not db_manager.database:
        raise RuntimeError("Database not initialized. Call connect_database() first.")
    return db_manager.database


async def get_async_session() -> AsyncSession:
    """Get an async SQLAlchemy session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_connection():
    """Context manager for database connections."""
    connection = await db_manager.get_connection()
    try:
        async with connection:
            yield connection
    except Exception as e:
        logger.error(f"Database connection error: {e}")
        raise


async def connect_database() -> None:
    """Initialize database connection."""
    await db_manager.connect()
    global database
    database = db_manager.database


async def disconnect_database() -> None:
    """Close database connection."""
    await db_manager.disconnect()
    global database
    database = None


async def test_database_connection() -> dict:
    """Test database connection and return status."""
    try:
        if not db_manager.is_connected:
            await connect_database()
        
        # Test basic connectivity
        health_ok = await db_manager.health_check()
        
        # Get database version using the databases package
        db = await get_database()
        version_result = await db.fetch_one("SELECT version() as version")
        
        # Get pool info if available
        pool_info = {
            "min_size": settings.database_min_connections,
            "max_size": settings.database_max_connections,
        }
        
        # Try to get pool stats if using asyncpg directly
        if hasattr(db_manager, '_connection_pool') and db_manager._connection_pool is not None:
            async with db_manager._connection_pool.acquire() as conn:
                pool_info.update({
                    "size": db_manager._connection_pool.get_size(),
                    "idle": db_manager._connection_pool.get_idle_size(),
                })
        
        return {
            "status": "connected" if health_ok else "error",
            "health_check": health_ok,
            "database_version": version_result["version"] if version_result else None,
            "pool_info": pool_info,
            "settings": {
                "database_name": settings.database_name,
                "host": settings.database_host,
                "port": settings.database_port,
            }
        }
        
    except Exception as e:
        logger.error(f"Database connection test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "health_check": False,
        }


# Dependency for FastAPI
async def get_db():
    """FastAPI dependency for database access."""
    return await get_database()


# Dependency for async sessions
async def get_session():
    """FastAPI dependency for SQLAlchemy async sessions."""
    async for session in get_async_session():
        yield session
