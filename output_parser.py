# takes the raw input and converts into python objects
# StrOutputParser, CommaSeparatedListOutputParser, JSONOutputParser, PydanticOutputParser

from pydantic import BaseModel
from llm_config import get_llm
from langchain_core.output_parsers import PydanticOutputParser, JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate

class Person(BaseModel):
    name: str
    age: int

llm = get_llm()

parser = JsonOutputParser(pydantic_object=Person)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are an AI assistant, who extract name and age from the given text"),
        ("human", "here is the text: {text}.\n {format_instructions}"),
    ]
).partial(format_instructions=parser.get_format_instructions())

chain = prompt | llm | parser

res =chain.invoke({"text":"My name is Mohit and age is 22"})

print(res)
