from pydantic_settings import BaseSettings


class Config(BaseSettings):
    openai_api_key: str
    openai_base_url: str = "https://api.openai.com/v1"

    agent_model: str = "gpt-4o-mini"
    agent_max_iterations: int = 20

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


config = Config()  # pyright: ignore[reportCallIssue]
