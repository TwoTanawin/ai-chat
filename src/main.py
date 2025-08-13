import logging
from .langchain.llm_model_loader import LLMLoader
from .langchain.embedding_model_loader import EmbeddingLoader
from .langchain.prompts.prompt import PromptRegistry
from .utils.aws.aws_config import get_bedrock_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class AIChat:
    def __init__(self):
        try:
            self.bedrock_client = get_bedrock_client()
            self.embeddings = EmbeddingLoader.load(client=self.bedrock_client)
            self.llm = LLMLoader.load(client=self.bedrock_client)
            self.prompt_registry = PromptRegistry()
        except Exception as e:
            logger.error(f"Error initializing PomGenerator: {e}")
            raise
        
    def runner(self):
        context = input("Ask : ")
        prompt = self.prompt_registry.get_prompt_templates("chat").format(
            human_input=context
        )
        
        result = self.llm.invoke(prompt)
        
        return result.content 

    
def main():
    try:
        ai_chat = AIChat()
        print(f"{ai_chat.runner()}")
    except Exception as e:
        print(f"Error : {e}")
        
if __name__=="__main__":
    main()