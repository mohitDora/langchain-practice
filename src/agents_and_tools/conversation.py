from dotenv import load_dotenv
from langchain import hub
from langchain.agents import (
    AgentExecutor,
    create_structured_chat_agent,
)
from langchain.memory import ConversationBufferMemory
from langchain_core.tools import Tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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

prompt = hub.pull("hwchase17/structured-chat-agent")

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

agent = create_structured_chat_agent(llm=llm, tools=tools, prompt=prompt)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, memory=memory
)

initial_message = "You are an AI assistant that can provide helpful answers using available tools.\nIf you are unable to answer, you can use the following tools: Time and Wikipedia."
memory.chat_memory.add_message(SystemMessage(content=initial_message))

while True:
    query = input("You: ")
    if query == "exit":
        break
    memory.chat_memory.add_message(HumanMessage(content=query))
    res = agent_executor.invoke({"input": query})
    memory.chat_memory.add_message(AIMessage(content=res["output"]))
    print(f"AI: ", res["output"])
