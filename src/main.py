import logging
import asyncio
from langchain_core.runnables.history import RunnableWithMessageHistory

from .langchain.llm_model_loader import LLMLoader
from .langchain.embedding_model_loader import EmbeddingLoader
from .langchain.prompts.prompt import PromptRegistry
from .utils.aws.aws_config import get_bedrock_client
from .langchain.memory import RedisMemory 

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

class AIChat:
    def __init__(self, session_id: str = "cli-session-1"):
        self.session_id = session_id
        self.bedrock_client = get_bedrock_client()
        self.embeddings = EmbeddingLoader.load(client=self.bedrock_client)
        self.llm = LLMLoader.load(client=self.bedrock_client)
        self.prompt_registry = PromptRegistry()

        chat_prompt = self.prompt_registry.get_prompt_templates("chat")
        base_chain = chat_prompt | self.llm

        self.memory = RedisMemory()

        def get_history(session_id: str):
            return self.memory.get_history(session_id)

        self.chat = RunnableWithMessageHistory(
            base_chain,
            get_history,
            input_messages_key="human_input",
            history_messages_key="history",
        )

    async def runner(self):
        user_input = await asyncio.to_thread(input, "Ask : ")
        user_input = user_input.strip()

        if not user_input:
            return ""

        ai_msg = await self.chat.ainvoke(
            {"human_input": user_input},
            config={"configurable": {"session_id": self.session_id}},
        )
        return ai_msg.content

async def main():
    ai_chat = AIChat(session_id="cli-session-1")
    try:
        while True:
            result = await ai_chat.runner()
            print(result)
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    except Exception as e:
        print(f"Error : {e}")

if __name__ == "__main__":
    asyncio.run(main())
