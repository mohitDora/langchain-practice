from pydantic import BaseModel

from langchain import hub
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import tool

from config.llm_config import get_llm


@tool()
def greet_user(name: str) -> str:
    """Useful for when you need to greet a user."""
    return "Hello" + name


class ConcatenateStrings(BaseModel):
    str1: str
    str2: str


@tool(args_schema=ConcatenateStrings)
def concatenate_strings(str1: str, str2: str) -> str:
    """Useful for when you want to concatenate two strings."""
    return str1 + str2


tools = [greet_user, concatenate_strings]

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
