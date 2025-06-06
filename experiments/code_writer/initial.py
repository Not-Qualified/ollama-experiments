from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM

# Load all source code files (adjust for your repo structure and file types)
loader = DirectoryLoader(
    path="/Users/pratikb/repos/ollama-experiments",
    loader_cls=TextLoader,
    glob=["*/*.py", "*.txt", "*.md"],  # Adjust file types as needed
    exclude=[
        "*/venv/**",
        "**/__pycache__/**",
    ],  # Exclude virtual environments and cache
)

documents = loader.load()

# Chunk into overlapping pieces
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=100,
    separators=["\nclass ", "\ndef ", "\n\n", "\n", " "],
)
docs = text_splitter.split_documents(documents)

print(len(docs))

for _ in docs:
    print(_.page_content)

print("step-1 done")


# Use Ollama for embedding
embedding = OllamaEmbeddings(
    model="nomic-embed-text:latest",
    num_ctx=9999,
)  # e.g., "nomic-embed-text"

# Create FAISS index
vectorstore = FAISS.from_documents(docs, embedding)

# Save index
# vectorstore.save_local("faiss_index_repo")
print("step-2 done")


# Load vector store
# vectorstore = FAISS.load_local("faiss_index_repo", embedding)
print("step-3 done")
# Setup retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# Use Ollama for LLM
llm = OllamaLLM(model="qwen3:4b", num_ctx=99999)  # e.g., "llama3", "codellama", etc.

qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# Ask questions about the code
response = qa_chain.invoke(
    "Create a langchain tool that can read all the source code files in a directory and chunk them into overlapping pieces, then create a vector store using Ollama embeddings and FAISS, and finally set up a retrieval chain to answer questions about the code."
)
print(response)
