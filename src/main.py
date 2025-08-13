import logging
from collections import defaultdict

from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from .langchain.llm_model_loader import LLMLoader
from .langchain.embedding_model_loader import EmbeddingLoader
from .langchain.prompts.prompt import PromptRegistry
from .utils.aws.aws_config import get_bedrock_client

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langchain_aws").setLevel(logging.WARNING)
logging.getLogger("botocore").setLevel(logging.WARNING)

class AIChat:
    def __init__(self, session_id: str = "cli-session-1"):
        try:
            self.session_id = session_id
            self.bedrock_client = get_bedrock_client()
            self.embeddings = EmbeddingLoader.load(client=self.bedrock_client)
            self.llm = LLMLoader.load(client=self.bedrock_client)
            self.prompt_registry = PromptRegistry()

            # Get your prompt (now includes a {history} slot)
            chat_prompt = self.prompt_registry.get_prompt_templates("chat")

            # Base chain
            base_chain = chat_prompt | self.llm

            # Per-session histories (RAM)
            self._histories = defaultdict(InMemoryChatMessageHistory)

            def get_history(session_id: str):
                return self._histories[session_id]

            # Wrap with memory
            self.chat = RunnableWithMessageHistory(
                base_chain,
                get_history,
                input_messages_key="human_input",  # matches the "{human_input}" variable
                history_messages_key="history",    # matches MessagesPlaceholder name
            )

        except Exception as e:
            logger.error(f"Error initializing AIChat: {e}")
            raise

    def runner(self):
        user_input = input("Ask : ").strip()
        if not user_input:
            return ""

        ai_msg = self.chat.invoke(
            {"human_input": user_input},
            config={"configurable": {"session_id": self.session_id}}
        )
        return ai_msg.content

def main():
    try:
        ai_chat = AIChat(session_id="cli-session-1")
        while True:
            print(ai_chat.runner())
    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    except Exception as e:
        print(f"Error : {e}")

if __name__=="__main__":
    main()
