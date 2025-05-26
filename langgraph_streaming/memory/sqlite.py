# Global Imports
from fastapi import FastAPI
from contextlib import asynccontextmanager
import aiosqlite
from typing import Optional
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


class SQLiteDBManager:
    """Singleton database manager for SQLite connections"""

    _instance: Optional['SQLiteDBManager'] = None
    _connection: Optional[aiosqlite.Connection] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SQLiteDBManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self, db_path: str = "./memory/langraph.sqlite"):
        """Initialize the database connection"""
        if self._connection is None:
            self._connection = await aiosqlite.connect(db_path)
            # Set up any required tables or configurations
            await self._setup_database()
        return self._connection

    async def _setup_database(self):
        """Set up database tables and configurations"""
        if self._connection:
            # Enable WAL mode for better concurrency
            await self._connection.execute("PRAGMA journal_mode=WAL")
            await self._connection.execute("PRAGMA synchronous=NORMAL")
            await self._connection.execute("PRAGMA cache_size=1000")
            await self._connection.execute("PRAGMA temp_store=memory")
            await self._connection.commit()

    def get_connection(self) -> Optional[aiosqlite.Connection]:
        """Get the current database connection"""
        if self._connection is None:
            raise RuntimeError(
                "Database not initialized. Call initialize() first.")
        return self._connection

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get a checkpointer instance using the managed connection"""
        if self._connection is None:
            raise RuntimeError(
                "Database not initialized. Call initialize() first.")
        return AsyncSqliteSaver(conn=self._connection)

    async def close(self):
        """Close the database connection"""
        if self._connection:
            await self._connection.close()
            self._connection = None

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._connection is not None


# Create global instance
sqlite_db_manager = SQLiteDBManager()

# Convenience functions for easy access


async def get_db_connection() -> aiosqlite.Connection:
    """Get database connection"""
    return sqlite_db_manager.get_coYeYeonnection()


def get_checkpointer() -> BaseCheckpointSaver:
    """Get checkpointer instance"""
    return sqlite_db_manager.get_checkpointer()


# main.py


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("Initializing database...")
    await sqlite_db_manager.initialize("./memory/langraph.sqlite")
    print("Database initialized successfully")

    yield

    # Shutdown
    print("Closing database connection...")
    await sqlite_db_manager.close()
    print("Database connection closed")

app = FastAPI(lifespan=lifespan)

# routes.py or any other module


async def some_function():
    """Example function using the database"""
    # Get checkpointer
    checkpointer = get_checkpointer()

    # Or get raw connection for custom queries
    conn = await get_db_connection()
    async with conn.execute("SELECT * FROM some_table") as cursor:
        rows = await cursor.fetchall()

    return rows


@app.get("/checkpoint")
async def create_checkpoint():
    """Example endpoint using checkpointer"""
    checkpointer = get_checkpointer()
    # Use checkpointer for your langraph operations
    return {"status": "checkpoint created"}


@app.get("/db-status")
async def db_status():
    """Check database status"""
    return {
        "initialized": sqlite_db_manager.is_initialized,
        "connection": str(sqlite_db_manager.get_connection()) if sqlite_db_manager.is_initialized else None
    }
