from __future__ import annotations

import os
from typing import Optional
from dotenv import load_dotenv
from langchain_community.chat_message_histories import RedisChatMessageHistory

load_dotenv()

def _int_or_none(value: Optional[str]) -> Optional[int]:
    try:
        return int(value) if value not in (None, "") else None
    except ValueError:
        return None
class RedisMemory:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_prefix = os.getenv("CHAT_HISTORY_PREFIX", "chat_history:")
        self.redis_ttl = _int_or_none(os.getenv("CHAT_HISTORY_TTL_SECONDS")) 
        
    def get_history(self, session_id: str)-> RedisChatMessageHistory:
        try:
            return RedisChatMessageHistory(
                    session_id=session_id,
                    url=self.redis_url,
                    key_prefix=self.redis_prefix,
                    ttl=self.redis_ttl,
            )
        except Exception as e:
            raise RuntimeError(f"Error : {e}")