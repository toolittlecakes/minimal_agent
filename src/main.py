import asyncio

from agent.core import Agent
from agent.session import ChatCompletionMessageParam, Session
from agent.tool import tool
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


async def ask(user_query: str) -> str:
    tools = [
        tool(CompanySpecificData(company_id="123").get_company_data),
        tool(get_weather),
    ]
    session = InMemorySession(session_id="42")
    usage_store = InMemoryUsageStore()

    agent = Agent(tools=tools, session=session, usage_store=usage_store)

    response = await agent.run(user_query)
    return response.reasoning + "\n" + response.answer


async def main():
    print(await ask("What is the weather around our office?"))


if __name__ == "__main__":
    asyncio.run(main())
