from langchain_community.document_loaders import TextLoader
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

current_dir = Path(__file__).resolve().parent.parent
books_dir = Path.joinpath(current_dir.parent, "books")
persistent_directory = Path.joinpath(current_dir.parent, "db", "chroma_db_with_metadata")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def ingest():
    if not os.path.exists(persistent_directory):
        print("Directory doesn't exist. Creating...")

        if not os.path.exists(books_dir):
            print(f"Folder doesn't exist. {books_dir}")
            raise FileNotFoundError("Folder doesn't exist")
        
        book_files= [f for f in os.listdir(books_dir) if f.endswith(".txt")]

        documents=[]

        for book in book_files:
            file_path = Path.joinpath(books_dir, book)
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = book
            documents.extend(docs)

        # print("Documents", documents)

        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        docs = text_splitter.split_documents(documents)

        # print("Docs", docs)

        vectorstore = Chroma.from_documents(
            docs, embedding_model, persist_directory=str(persistent_directory)
        )

        print("Vectorstore created")
    else:
        print("Directory already exists")

def retrieve():
    db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

    query = "Who is Odysseus's wife?"

    retriever = db.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 10, "score_threshold": 0.5},
    )

    relevant_docs = retriever.invoke(query)

    print("\n--- Relevant Documents ---")
    for i, doc in enumerate(relevant_docs, 1):
        print(f"Document {i}:\n{doc.page_content}\n")
        if doc.metadata:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}\n")

if __name__ == "__main__":
    ingest()
    retrieve()