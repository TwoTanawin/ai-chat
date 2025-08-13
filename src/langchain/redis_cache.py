from dotenv import load_dotenv
import os

load_dotenv()

class RedisCache:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_prefix = os.getenv("CHAT_HISTORY_PREFIX", "chat_history:")
