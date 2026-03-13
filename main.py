import getpass
import os
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = getpass.getpass()
if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

# Google Gemini chat model
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")

# Google Gemini embeddings model
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

# FAISS vector store
embedding_dim = len(embeddings.embed_query("hello world"))
index = faiss.IndexFlatL2(embedding_dim)

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


def main():
    print("Hello from rag-agent!")


if __name__ == "__main__":
    main()
