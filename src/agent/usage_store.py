from typing import Protocol

from openai.types.completion_usage import CompletionUsage


class UsageStore(Protocol):
    async def add_usage(self, usage: CompletionUsage): ...
