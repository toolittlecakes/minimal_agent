import inspect
from functools import wraps
from typing import Callable

from openai.types.chat.chat_completion_function_tool_param import (
    ChatCompletionFunctionToolParam,
)
from pydantic import TypeAdapter


def function_to_schema(f) -> ChatCompletionFunctionToolParam:
    schema = TypeAdapter(f).json_schema()
    return {
        "type": "function",
        "function": {
            "name": f.__name__,
            "description": f.__doc__,
            "parameters": schema,
            "strict": True,
        },
    }


class Tool[**P, R]:
    """Wrapper for a function with a json_schema. Works with both sync and async functions."""

    def __init__(self, func: Callable[P, R]) -> None:
        self._func = func
        self._is_async = inspect.iscoroutinefunction(func)
        wraps(func)(self)

    @property
    def is_async(self) -> bool:
        """Returns True if the wrapped function is async."""
        return self._is_async

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Call the wrapped function. Returns a coroutine if async, result if sync."""
        return self._func(*args, **kwargs)

    @property
    def schema(self) -> ChatCompletionFunctionToolParam:
        return function_to_schema(self._func)


def tool[**P, R](func: Callable[P, R]) -> Tool[P, R]:
    return Tool[P, R](func)


if __name__ == "__main__":
    # example of using the tool decorator
    
    import asyncio

    @tool
    def foo(a: int, b: int) -> int:
        """Adds two integers together"""
        return a + b

    @tool
    async def bar(a: int, b: int) -> int:
        """Multiplies two integers asynchronously"""
        return a * b

    class Api:
        def __init__(self, base_url: str):
            self.base_url = base_url

        async def get_user(self, user_id: int) -> dict:
            """Gets a user by ID"""
            return {"id": user_id, "name": "John Doe"}

    async def main():
        print("Sync tool:", foo.is_async, foo(1, 2))
        print("Async tool:", bar.is_async, await bar(3, 4))
        print("Async schema:", bar.schema)

        api = Api("https://api.example.com")
        get_user = tool(api.get_user)
        print("Async tool:", get_user.is_async, await get_user(1))
        print("Async schema:", get_user.schema)

    asyncio.run(main())
