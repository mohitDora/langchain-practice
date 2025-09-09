from pathlib import Path
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings

load_dotenv()

current_dir = Path(__file__).resolve().parent.parent
persistent_directory = Path.joinpath(current_dir.parent, "db", "chroma_db")

embedding_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

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
