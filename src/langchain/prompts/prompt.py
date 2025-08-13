# prompts/prompt.py
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class PromptRegistry:
    def __init__(self):
        self.prompt_templates = {
            "chat": ChatPromptTemplate.from_messages([
                ("system",
                 "You are a concise, helpful assistant. "
                 "Answer clearly. If unsure, ask for clarification."
                ),
                # <-- This is where the running conversation will be injected
                MessagesPlaceholder(variable_name="history"),
                ("human", "{human_input}"),
            ])
        }

    def get_prompt_templates(self, task_type: str) -> ChatPromptTemplate:
        if task_type not in self.prompt_templates:
            raise ValueError(f"Unsupported task type: {task_type}")
        return self.prompt_templates[task_type]
