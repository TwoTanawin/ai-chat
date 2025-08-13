import logging
import os
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingLoader:
    _embeddings_model = None

    @classmethod
    def load(cls, model_id=None, client=None):
        if cls._embeddings_model is None:
            if model_id is None:
                model_id = os.getenv('EMBEDDING_MODEL')
            if not model_id:
                raise ValueError("Embedding model ID not provided and EMBEDDING_MODEL not set in environment.")

            logger.info(f"Loading embedding model {model_id}...")
            cls._embeddings_model = BedrockEmbeddings(
                model_id=model_id,
                client=client
            )
            logger.info("Embedding model loaded successfully.")
        return cls._embeddings_model
