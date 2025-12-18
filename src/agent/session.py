from typing import Protocol
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

class Session(Protocol):
    async def add_message(self, message: ChatCompletionMessageParam):
        ...

    async def get_messages(self) -> list[ChatCompletionMessageParam]:
        ...
