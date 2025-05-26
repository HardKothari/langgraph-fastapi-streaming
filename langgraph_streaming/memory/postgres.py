from typing import Optional
import logging
from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

logger = logging.getLogger(__name__)


class PostgresDBManager:
    """Singleton database manager for PostgreSQL connections using connection pool"""

    _instance: Optional['PostgresDBManager'] = None
    _connection_pool: Optional[AsyncConnectionPool] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PostgresDBManager, cls).__new__(cls)
        return cls._instance

    async def initialize(self, connection_string: str, max_size: int = 20):
        """Initialize the database connection pool"""
        if self._connection_pool is None:
            connection_kwargs = {
                "autocommit": True,
                "prepare_threshold": 0,
            }

            self._connection_pool = AsyncConnectionPool(
                conninfo=connection_string,
                max_size=max_size,
                open=False,
                kwargs=connection_kwargs
            )

            # Open the connection pool
            await self._connection_pool.open(wait=True)

            # Set up database tables and configurations
            await self._setup_database()

            logger.info("PostgreSQL connection pool initialized successfully")

        return self._connection_pool

    async def _setup_database(self):
        """Set up database tables and configurations"""
        if self._connection_pool:
            try:
                async with self._connection_pool.connection() as conn:
                    await AsyncPostgresSaver(conn).setup()
                logger.info("Database setup completed successfully")
            except Exception as e:
                logger.exception(
                    f"Error setting up database: {type(e).__name__} - {e}")
                raise

    def get_connection_pool(self) -> AsyncConnectionPool:
        """Get the current database connection pool"""
        if self._connection_pool is None:
            raise RuntimeError(
                "Database not initialized. Call initialize() first.")
        return self._connection_pool

    def get_checkpointer(self) -> BaseCheckpointSaver:
        """Get a checkpointer instance using the managed connection pool"""
        if self._connection_pool is None:
            raise RuntimeError(
                "Database not initialized. Call initialize() first.")
        return AsyncPostgresSaver(conn=self._connection_pool)  # type: ignore

    async def close(self):
        """Close the database connection pool"""
        if self._connection_pool:
            await self._connection_pool.close()
            self._connection_pool = None
            logger.info("PostgreSQL connection pool closed")

    @property
    def is_initialized(self) -> bool:
        """Check if database is initialized"""
        return self._connection_pool is not None


# Create global instance
postgres_db_manager = PostgresDBManager()


# Convenience functions for easy access
async def initialize_database_connection_pool(connection_string: str, max_size: int = 20):
    """Initialize database connection pool"""
    return await postgres_db_manager.initialize(connection_string, max_size)


def get_connection_pool() -> AsyncConnectionPool:
    """Get database connection pool"""
    return postgres_db_manager.get_connection_pool()


def get_checkpointer() -> BaseCheckpointSaver:
    """Get checkpointer instance"""
    return postgres_db_manager.get_checkpointer()


async def close_connection_pool():
    """Close database connection pool"""
    await postgres_db_manager.close()
