from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from config.llm_config import get_llm

llm = get_llm()

template = "Tell me {number_of_jokes} jokes on {topic}"

prompt = PromptTemplate(template=template, input_variables=["number_of_jokes", "topic"])

formatted_prompt = prompt.format(number_of_jokes="3", topic="cats")

res = llm.invoke(formatted_prompt)

print(formatted_prompt)
print(res.content)

messages = [
    ("system", "You are comedian who tells jokes on {topic}"),
    ("human", "Tell me {number_of_jokes} jokes"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.format(number_of_jokes="3", topic="cats")

res = llm.invoke(prompt)
print(res.content)