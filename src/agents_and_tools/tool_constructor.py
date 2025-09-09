from pydantic import BaseModel

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import StructuredTool, Tool

from config.llm_config import get_llm


def greet_user(name: str) -> str:
    return "Hello" + name


def concatenate_strings(str1: str, str2: str) -> str:
    return str1 + str2


class ConcatenateStrings(BaseModel):
    str1: str
    str2: str


tools = [
    Tool(
        name="Greet User",
        func=greet_user,
        description="Useful for when you need to greet a user.",
    ),
    StructuredTool.from_function(
        func=concatenate_strings,
        name="Concatenate Strings",
        description="Useful for when you want to concatenate two strings.",
        args_schema=ConcatenateStrings,
    ),
]

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
