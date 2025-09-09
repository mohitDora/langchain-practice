from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_react_agent,
)
from langchain_core.tools import Tool
from config.llm_config import get_llm

import datetime

# Load environment variables from .env file
load_dotenv()


def current_time(*args, **kwargs):
    return "The current time is " + str(datetime.datetime.now())


tools = [
    Tool(
        name="Current Time",
        func=current_time,
        description="Useful for when you need to know the current time.",
    )
]

llm = get_llm()

prompt = hub.pull("hwchase17/react")

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True
)

res = agent_executor.invoke({"input": "What is the current time?"})

print(res)
