#!/usr/bin/env python3
"""
Complete database setup script for Voice Assistant AI Backend.
This script handles database creation, migrations, and data seeding.
"""

import asyncio
import sys
import os
import subprocess
from pathlib import Path

# Add the app directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from app.config import settings
from app.db.init_db import initialize_database, test_database_operations
from app.db.seed_data import seed_database, verify_database


async def run_alembic_upgrade():
    """Run Alembic migrations."""
    try:
        print("🔄 Running database migrations...")
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print("✅ Database migrations completed successfully!")
            return True
        else:
            print(f"❌ Migration failed: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False


async def setup_complete_database():
    """Complete database setup process."""
    print("🚀 Starting Voice Assistant AI Database Setup")
    print("=" * 60)
    
    # Step 1: Initialize database connection and create tables
    print("\n📋 Step 1: Initializing database...")
    try:
        success = await initialize_database()
        if not success:
            print("❌ Database initialization failed!")
            return False
        print("✅ Database initialized successfully!")
    except Exception as e:
        print(f"❌ Database initialization error: {e}")
        return False
    
    # Step 2: Run Alembic migrations
    print("\n📋 Step 2: Running migrations...")
    migration_success = await run_alembic_upgrade()
    if not migration_success:
        print("⚠️  Migration failed, but continuing with setup...")
    
    # Step 3: Test database operations
    print("\n📋 Step 3: Testing database operations...")
    try:
        test_results = await test_database_operations()
        if not all([
            test_results.get("connection_test", False),
            test_results.get("query_test", False),
            test_results.get("transaction_test", False)
        ]):
            print("❌ Database tests failed!")
            print(f"Test results: {test_results}")
            return False
        print("✅ Database tests passed!")
    except Exception as e:
        print(f"❌ Database testing error: {e}")
        return False
    
    # Step 4: Seed sample data
    print("\n📋 Step 4: Seeding sample data...")
    try:
        success = await seed_database(clear_existing=True)
        if not success:
            print("❌ Data seeding failed!")
            return False
        print("✅ Sample data seeded successfully!")
    except Exception as e:
        print(f"❌ Data seeding error: {e}")
        return False
    
    # Step 5: Verify final setup
    print("\n📋 Step 5: Verifying setup...")
    try:
        counts = await verify_database()
        print("📊 Final Database Contents:")
        for table, count in counts.items():
            print(f"  {table}: {count} records")
        
        if all(count > 0 for count in counts.values()):
            print("✅ Database verification successful!")
        else:
            print("⚠️  Some tables are empty, but setup completed.")
            
    except Exception as e:
        print(f"❌ Database verification error: {e}")
        return False
    
    print("\n🎉 Database setup completed successfully!")
    print("\n📝 Next steps:")
    print("  1. Start your FastAPI server: uvicorn app.main:app --reload")
    print("  2. Access API docs at: http://localhost:8000/docs")
    print("  3. Test the customer support endpoints")
    
    return True


async def quick_test():
    """Quick database connectivity test."""
    print("🔍 Quick Database Test")
    print("=" * 30)
    
    try:
        from app.db.connection import test_database_connection
        result = await test_database_connection()
        
        if result["status"] == "connected":
            print("✅ Database connection: OK")
            print(f"✅ Health check: {'PASSED' if result['health_check'] else 'FAILED'}")
            if "database_version" in result:
                print(f"✅ PostgreSQL version: {result['database_version']}")
            return True
        else:
            print(f"❌ Database connection failed: {result.get('error', 'Unknown error')}")
            return False
            
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False


def print_usage():
    """Print usage information."""
    print("Voice Assistant AI Database Setup")
    print("=" * 40)
    print("Usage: python setup_database.py [command]")
    print("\nCommands:")
    print("  setup    - Complete database setup (default)")
    print("  test     - Quick connectivity test")
    print("  migrate  - Run migrations only")
    print("  seed     - Seed data only")
    print("  verify   - Verify database contents")
    print("\nExamples:")
    print("  python setup_database.py setup")
    print("  python setup_database.py test")


async def main():
    """Main CLI function."""
    command = sys.argv[1].lower() if len(sys.argv) > 1 else "setup"
    
    try:
        if command == "setup":
            success = await setup_complete_database()
            sys.exit(0 if success else 1)
            
        elif command == "test":
            success = await quick_test()
            sys.exit(0 if success else 1)
            
        elif command == "migrate":
            success = await run_alembic_upgrade()
            sys.exit(0 if success else 1)
            
        elif command == "seed":
            print("🌱 Seeding database...")
            success = await seed_database(clear_existing=True)
            if success:
                print("✅ Database seeded successfully!")
            else:
                print("❌ Database seeding failed!")
            sys.exit(0 if success else 1)
            
        elif command == "verify":
            print("🔍 Verifying database...")
            counts = await verify_database()
            print("📊 Database Contents:")
            for table, count in counts.items():
                print(f"  {table}: {count} records")
            sys.exit(0)
            
        elif command in ["help", "-h", "--help"]:
            print_usage()
            sys.exit(0)
            
        else:
            print(f"Unknown command: {command}")
            print_usage()
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n⚠️  Setup interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Setup error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
