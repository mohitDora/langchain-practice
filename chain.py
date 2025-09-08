# anything that can be called with input and returns ouptut - Runnables
# RunnableLambada wraps any function into a runnable

from llm_config import get_llm
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema.runnable import (
    RunnableLambda,
    RunnableSequence,
    RunnableParallel,
    RunnableBranch,
)
from langchain.prompts import PromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant"),
        ("human", "{text}"),
    ]
)

#sequential chains
chain = prompt | get_llm() | StrOutputParser()

# res = chain.invoke({"text": "What is 2+2?"})
# print(res)

# chain under the hood

format_prompt = RunnableLambda(lambda x: prompt.format(text=x))
invoke_model = RunnableLambda(lambda x: get_llm().invoke(x))
parse_output = RunnableLambda(lambda x: x.content)

chain = RunnableSequence(first=format_prompt, middle=[invoke_model], last=parse_output)

# res = chain.invoke({"text": "What is 2+2?"})
# print(res)

messages = [
    ("system", "You are comedian who tells jokes on {topic}"),
    ("human", "Tell me {number_of_jokes} jokes"),
]

prompt_template = ChatPromptTemplate.from_messages(messages)

uppercase = RunnableLambda(lambda x: x.upper())
countWords = RunnableLambda(lambda x: len(x.split(" ")))

chain = prompt_template | get_llm() | StrOutputParser() | uppercase | countWords

# res = chain.invoke({"number_of_jokes": "3", "topic": "cats"})
# print(res)

# parallel chains

prompt1 = PromptTemplate.from_template("Translate {text} into hindi")
prompt2 = PromptTemplate.from_template("Translate {text} into french")
prompt3 = PromptTemplate.from_template("Translate {text} into spanish")

chain1 = prompt1 | get_llm() | StrOutputParser()
chain2 = prompt2 | get_llm() | StrOutputParser()
chain3 = prompt3 | get_llm() | StrOutputParser()

chain = RunnableParallel(hindi=chain1, french=chain2, spanish=chain3)

# res = chain.invoke({"text": "Hello"})
# print(res)

# branching chains

positive_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant"),
        ("human", "Generate a thanks message for {feedback}"),
    ]
)

negative_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant"),
        ("human", "Generate a apology message for {feedback}"),
    ]
)

neutral_feedback = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant"),
        ("human", "Generate a message for more detail about {feedback}"),
    ]
)

classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are an AI assistant"),
        (
            "human",
            "Please classify this feedback as positive, negative or neutral: {feedback}",
        ),
    ]
)

branches = RunnableBranch(
    (lambda x: "positive" in x.lower(), positive_feedback | get_llm() | StrOutputParser()),
    (lambda x: "negative" in x.lower(), negative_feedback | get_llm() | StrOutputParser()),
    neutral_feedback | get_llm() | StrOutputParser()
)

classification_chain = classification_template | get_llm() | StrOutputParser()

chain = classification_chain | branches

res = chain.invoke({"feedback": "This is a bad movie"})
print(res)
