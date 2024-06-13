import redis
from redis.cluster import RedisCluster
import logging
import os
from .config import settings

logger = logging.getLogger(__name__)

class RedisManager:
    def __init__(self, host="localhost", port=6379, password=None):
        self.host = host
        self.port = port
        self.password = password
        self.redis_connection = None

    async def ensure_connection(self):
        if self.redis_connection is None:
            try:
                if os.getenv("DEBUG", None) == "True" or os.getenv("SELF_HOSTING", None) == "True":
                    # Standalone Redis for development
                    self.redis_connection = redis.Redis(
                        host=self.host,
                        port=self.port,
                        password=self.password
                    )
                else:
                    # Redis Cluster for production or other environments
                    self.redis_connection = RedisCluster.from_url(
                        f"rediss://:{self.password}@{self.host}:{self.port}"
                        )
                    logger.info(f"Connected to Redis: {self.redis_connection.get_nodes()}")
            except Exception as e:
                logger.error(f"Could not connect to Redis: {e}")
                raise

    async def startup(self):
        await self.ensure_connection()

    async def shutdown(self):
        if self.redis_connection:
            self.redis_connection.connection_pool.disconnect()

    async def set_value(self, key, value):
        await self.ensure_connection()
        self.redis_connection.set(key, value)

    async def get_value(self, key: str):
        await self.ensure_connection()
        return self.redis_connection.get(key)

# Instantiate the RedisManager
redis_manager = RedisManager(
    host=settings.redis_host,
    port=int(settings.redis_port),
    password=settings.redis_password,
)
