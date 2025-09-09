from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

current_dir = Path(__file__).resolve().parent.parent
file_path = Path.joinpath(current_dir.parent, "books", "book1.txt")
persistent_directory = Path.joinpath(current_dir.parent, "chroma_db")

if not os.path.exists(persistent_directory):
    print("Directory doesn't exist. Creating...")

    if not os.path.exists(file_path):
        print(f"File doesn't exist. {file_path}")
        raise FileNotFoundError("File doesn't exist")

    loader = TextLoader(str(file_path), encoding="utf-8")
    documents = loader.load()

    # print("Documents", documents)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs = text_splitter.split_documents(documents)

    # print("Docs", docs)

    embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    vectorstore = Chroma.from_documents(
        docs, embedding_model, persist_directory=str(persistent_directory)
    )

    print("Vectorstore created")
else:
    print("Directory already exists")
