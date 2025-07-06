from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain.embeddings.ollama import OllamaEmbeddings

# file_path = "https://www.cgsmedicare.com/pdf/edi/835_compguide.pdf"
file_path = "../../../../Downloads/835_compguide.pdf"
loader = PyPDFLoader(file_path)

docs = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)

print(len(all_splits))

ollama_embedding = OllamaEmbeddings(model='mxbai-embed-large:latest')

vec1 = ollama_embedding.embed_query(all_splits[0].page_content)

print(vec1)