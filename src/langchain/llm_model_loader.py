import logging
import os
from langchain_aws import ChatBedrockConverse

logger = logging.getLogger(__name__)

class LLMLoader:
    _llm_model = None

    @classmethod
    def load(cls, model_id=None, client=None, temperature=0.8, max_tokens=2048):
        if cls._llm_model is None:
            if model_id is None:
                model_id = os.getenv("LLM_MODEL")
            if not model_id:
                raise ValueError("LLM model ID not provided and LLM_MODEL_ID not set in environment.")
            if client is None:
                raise ValueError("Bedrock client must be provided to load LLM.")

            logger.info(f"Loading LLM model {model_id}...")
            cls._llm_model = ChatBedrockConverse(
                model=model_id,
                client=client,
                temperature=temperature,
                max_tokens=max_tokens
            )
            logger.info("LLM model loaded successfully.")
        return cls._llm_model
