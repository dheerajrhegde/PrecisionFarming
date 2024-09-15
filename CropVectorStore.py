import re

from bs4 import BeautifulSoup

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import RecursiveUrlLoader, WebBaseLoader, PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_openai import OpenAIEmbeddings

class CropVectorStore:
    def bs4_extractor(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        return re.sub(r"\n\n+", "\n\n", soup.text).strip()

    def create_vector_store(self):
        loader = PyPDFLoader("guides/soybean.pdf") #RecursiveUrlLoader("https://soybeans.ces.ncsu.edu/", extractor=self.bs4_extractor)
        docs = loader.load()
        print("sybeans.ces.ncsu.edu", len(docs))
        loader = PyPDFLoader("guides/corn.pdf") #RecursiveUrlLoader("https://corn.ces.ncsu.edu/", extractor=self.bs4_extractor)
        docs = docs + loader.load()
        print("sybeans.ces.ncsu.edu + corn.ces.ncsu.edu", len(docs))
        loader = PyPDFLoader("guides/cotton.pdf") #RecursiveUrlLoader("https://cotton.ces.ncsu.edu/", extractor=self.bs4_extractor)
        docs = docs + loader.load()
        print("sybeans.ces.ncsu.edu + corn.ces.ncsu.edu + cottton.ces.ncsu.edu", len(docs))


        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1024, chunk_overlap=128
        )
        doc_splits = text_splitter.split_documents(docs)
        print("Splits done...", len(doc_splits))

        # Add to vectorDB
        vector_store = Chroma(
            collection_name="agriculture",
            embedding_function=OpenAIEmbeddings(),
            persist_directory="./chroma_langchain_db",  # Where to save data locally, remove if not neccesary
        )

        vector_store.add_documents(filter_complex_metadata(doc_splits))
        vector_store.persist()
        print("Vectorstore created...")

if __name__ == "__main__":
    cvs = CropVectorStore()
    cvs.create_vector_store()
