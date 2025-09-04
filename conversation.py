from llm_config import get_llm
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

llm = get_llm()

messages = []

messages.append(SystemMessage(content="You are an AI assistant"))

while True:
    query = input("You: ")
    if query == "exit":
        break
    messages.append(HumanMessage(content=query))

    res = llm.invoke(messages)
    messages.append(AIMessage(content=res.content))

    print(f"AI: ", res.content)