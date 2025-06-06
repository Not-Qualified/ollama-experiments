import sqlite3

from langchain_ollama import OllamaEmbeddings

from langchain_postgres import PGVector



from re import sub
from bs4 import BeautifulSoup
from langchain_community.document_loaders import RecursiveUrlLoader



def bs4_extractor(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    return sub(r"\n\n+", "\n\n", soup.text).strip()


loader = RecursiveUrlLoader(
    "https://python.langchain.com/",
    prevent_outside=True,
    base_url="https://python.langchain.com/",
    extractor=bs4_extractor,
)
#TODO: use HTMLSemanticPreservingSplitter
docs = loader.load()



# See docker command above to launch a postgres instance with pgvector enabled.
connection = "postgresql+psycopg://langchain:langchain@localhost:5432/langchain"  # Uses psycopg3!
collection_name = "langchain_docs"

# Use Ollama for embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest",num_ctx=9999,)  # e.g., "nomic-embed-text"
# TODO: use multiple models for embeddings, e.g., "nomic-embed-text", "nomic-embed-text:latest", etc. and rerank them.


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=collection_name,
    connection=connection,
    use_jsonb=True,
    create_extension=True,
)



stored_docs = vector_store.add_documents(docs,)  #ids=[hash(doc.page_content) for doc in docs])

print(f"Stored {len(stored_docs)} documents in the vector store.")