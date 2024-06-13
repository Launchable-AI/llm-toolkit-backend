import os
from typing import Optional

from slowapi import Limiter
from slowapi.util import get_remote_address

from pydantic import BaseSettings

class Settings(BaseSettings):
    debug: bool = False
    redis_host: str = "redis"
    redis_port: int = 6379
    redis_password: Optional[str] = None
    #redis_max_connections: int = 20
    jwt_secret_key: str
    
    class Config:
        env_file = ".dev.env"


# Initialize settings object
settings = Settings()

front_end_limiter = Limiter(
    key_func=get_remote_address,
    storage_uri=f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}"
)

back_end_limiter = Limiter(
    key_func=lambda request: request.headers.get("Bubble-Caller-Id"),
    storage_uri=f"redis://:{settings.redis_password}@{settings.redis_host}:{settings.redis_port}"
)
