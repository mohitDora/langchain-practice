from pydantic import BaseModel
from typing import Type

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import BaseTool

from config.llm_config import get_llm


class ConcatenateStringsInput(BaseModel):
    str1: str
    str2: str


class GreetUserInput(BaseModel):
    str1: str


class ConcatenateStrings(BaseTool):
    name: str = "Concatenate Strings"
    description: str = "Useful for when you want to concatenate two strings."
    args_schema: Type[BaseModel] = ConcatenateStringsInput

    def _run(self, str1: str, str2: str) -> str:
        return str1 + str2

    async def _arun(self, str1: str, str2: str) -> str:
        raise NotImplementedError


class GreetUser(BaseTool):
    name: str = "Greet User"
    description: str = "Useful for when you need to greet a user."
    args_schema: Type[BaseModel] = GreetUserInput

    def _run(self, str1: str) -> str:
        return f"Hello {str1}"

    async def _arun(self, str1: str) -> str:
        raise NotImplementedError


tools = [GreetUser(), ConcatenateStrings()]

llm = get_llm()

prompt = hub.pull("hwchase17/openai-tools-agent")

agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

response = agent_executor.invoke({"input": "Greet Alice"})
print("Response for 'Greet Alice':", response)

response = agent_executor.invoke({"input": "Concatenate 'hello' and 'world'"})
print("Response for 'Concatenate hello and world':", response)
