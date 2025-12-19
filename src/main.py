import asyncio

from pydantic import BaseModel, Field

from agent.core import Agent
from agent.session import ChatCompletionMessageParam, Session
from agent.tool import Tool
from agent.usage_store import CompletionUsage, UsageStore


# TOOLS
async def get_weather(city: str) -> dict:
    return {
        "city": city,
        "weather": "sunny",
    }


class CompanySpecificData:
    def __init__(self, company_id: str):
        self.company_id = company_id

    async def get_company_data(self) -> dict:
        """
        Get the company data of the user
        """
        return {
            "company_id": self.company_id,
            "company_name": "Company Name",
            "company_address": "123 Main St, Anytown, USA",
        }


# SESSION STORAGE
class InMemorySession(Session):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.messages: list[ChatCompletionMessageParam] = []

    async def add_message(self, message: ChatCompletionMessageParam):
        print(message)
        print("-" * 20)
        self.messages.append(message)

    async def get_messages(self) -> list[ChatCompletionMessageParam]:
        return self.messages


class InMemoryUsageStore(UsageStore):
    async def add_usage(self, usage: CompletionUsage):
        # print(usage)
        ...


class AgentResponse(BaseModel):
    """Call this tool only when you have all the information for a response."""

    reasoning: list[str] = Field(
        ...,
        description="Stating the retrieved facts relevant to the question asked.",
        min_length=1,
        max_length=4,
    )
    answer: str


async def ask(user_query: str):
    tools = [
        Tool(CompanySpecificData(company_id="123").get_company_data),
        Tool(get_weather),
    ]
    session = InMemorySession(session_id="42")
    usage_store = InMemoryUsageStore()

    agent = Agent(
        tools=tools,
        session=session,
        usage_store=usage_store,
        response_model=AgentResponse,
    )

    response = await agent.run(user_query)
    return response


async def main():
    response = await ask("What is the weather around our office?")
    print("Final response:")
    print(response)


if __name__ == "__main__":
    asyncio.run(main())
