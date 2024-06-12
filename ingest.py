# import libraries for ingestion
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

# Get embeddings for converting the format into computer-readable format
embeddings = SentenceTransformerEmbeddings(model_name='all-mpnet-base-v2')

# Print the embeddings
print(embeddings)

# Load pdf files from the directory
loader = DirectoryLoader('data/', glob='**/*.pdf', show_progress=True, loader_cls=PyPDFLoader)

# store loaded documents in a variable
documents = loader.load()

# Split the text into reasonable chunk sizes
splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=500)

# Split the text into chunks and store them in a variable
texts = splitter.split_documents(documents)

# Print the texts
print(texts)


# Create vector storage using Chroma
vector_store = Chroma.from_documents(texts, embeddings, collection_metadata={"hnsw:space": "cosine"}, persist_directory="storage/microplastic_cosine")

# Print the vector store completion
print("Vector DB created.")