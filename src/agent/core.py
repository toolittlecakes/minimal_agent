import json

from openai import AsyncOpenAI
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionAssistantMessageParam,
    ChatCompletionMessageFunctionToolCallParam,
    ChatCompletionMessageToolCallUnion,
    ChatCompletionToolMessageParam,
)
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam
from openai.types.chat.chat_completion_message_tool_call_param import Function
from pydantic import BaseModel, Field

from agent.session import Session
from agent.tool import Tool
from agent.usage_store import UsageStore
from config import config

client = AsyncOpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url)


def _convert_tool_calls_to_message(
    tool_calls: list[ChatCompletionMessageToolCallUnion],
) -> ChatCompletionAssistantMessageParam:
    """Utility function for type checking."""
    return ChatCompletionAssistantMessageParam(
        role="assistant",
        tool_calls=[
            ChatCompletionMessageFunctionToolCallParam(
                id=tool_call.id,
                function=Function(
                    name=tool_call.function.name, arguments=tool_call.function.arguments
                ),
                type="function",
            )
            for tool_call in tool_calls
            if tool_call.type == "function"
        ],
    )


async def _make_generation_step(
    messages: list[ChatCompletionMessageParam],
    tool_by_name: dict[str, Tool],
    model: str = config.agent_model,
) -> ChatCompletion:
    tools_schemas = [tool.schema for tool in tool_by_name.values()]
    return await client.chat.completions.create(
        model=model,
        tools=tools_schemas,
        tool_choice="required",
        messages=messages,
    )


class DefaultResponse(BaseModel):
    """Call this function to return a final response from the agent."""

    answer: str = Field(description="The final answer to the question.", min_length=1)


class Agent[R: BaseModel = DefaultResponse]:
    def __init__(
        self,
        tools: list[Tool],
        session: Session,
        usage_store: UsageStore,
        response_model: type[R] = DefaultResponse,
    ):
        # self.response_tool = response_model_to_tool(response_model)
        self.response_tool = Tool(response_model)
        tools = [*tools, self.response_tool]
        # check that tools are unique
        self.tool_by_name: dict[str, Tool] = {
            tool.schema["function"]["name"]: tool for tool in tools
        }
        if len(self.tool_by_name) != len(tools):
            raise ValueError("Tools must be unique")

        self.session = session
        self.usage_store = usage_store

    async def run(self, prompt: str) -> R:
        system_prompt = "you can use tools to answer questions"
        await self.session.add_message({"role": "system", "content": system_prompt})
        await self.session.add_message({"role": "user", "content": prompt})

        # NOTE: we can use local messages variable that mirrors the session state to avoid multiple calls to get_messages

        for i in range(config.agent_max_iterations):
            response = await _make_generation_step(
                await self.session.get_messages(), self.tool_by_name
            )
            # store usage
            if response.usage:
                await self.usage_store.add_usage(response.usage)

            tool_calls = response.choices[0].message.tool_calls
            if not tool_calls:
                raise ValueError("No tool calls found")

            # Add model response to session
            await self.session.add_message(_convert_tool_calls_to_message(tool_calls))

            for tool_call in tool_calls:
                if tool_call.type != "function":
                    continue

                tool = self.tool_by_name[tool_call.function.name]
                arguments = json.loads(tool_call.function.arguments)

                result = await tool(**arguments) if tool.is_async else tool(**arguments)
                if tool == self.response_tool:
                    return result

                message = ChatCompletionToolMessageParam(
                    role="tool",
                    tool_call_id=tool_call.id,
                    content=str(result),
                )
                await self.session.add_message(message)

        raise ValueError(f"No response after {config.agent_max_iterations} iterations")
