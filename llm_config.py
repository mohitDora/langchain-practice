from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "openai/gpt-oss-120b"

def get_llm():
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=1
    )
    return llm