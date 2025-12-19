import inspect
from functools import wraps
from typing import Callable

from openai.types.chat.chat_completion_function_tool_param import (
    ChatCompletionFunctionToolParam,
)
from pydantic import BaseModel, TypeAdapter


def function_to_schema[**P, R](f: Callable[P, R]) -> ChatCompletionFunctionToolParam:
    """Handles both regular functions and Pydantic models."""
    schema = TypeAdapter(f).json_schema()

    if isinstance(f, type) and issubclass(f, BaseModel):
        fields_to_remove = ("name", "description", "title")
        schema = {k: v for k, v in schema.items() if k not in fields_to_remove}
        schema = schema | {"additionalProperties": False}

    return {
        "type": "function",
        "function": {
            "name": f.__name__,
            "description": f.__doc__ or "",
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
