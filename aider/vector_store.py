from langchain_community.embeddings import SentenceTransformerEmbeddings, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import SQLiteVSS
import os
from langchain_community.vectorstores import SQLiteVSS
from langchain_text_splitters import CharacterTextSplitter

from openai import api_key


class VectorStore:
    def __init__(self):
        self.vector_store = SQLiteVSS(
            table="aider_vectors",
            connection=SQLiteVSS.create_connection(db_file="/tmp/vss.db"),
            embedding=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        )
        self.text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)

    def get_db_stats(self):
        """Get database statistics."""
        cursor = self.vector_store._connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM aider_vectors")
        count = cursor.fetchone()[0]
        return f"Total entries in vector store: {count}"

    def add_text(self, text, metadata=None):
        """Add text to the SQLiteVSS vector store."""
        chunks = self.text_splitter.split_text(text)
        metadatas = [metadata] * len(chunks) if metadata else None
        self.vector_store.add_texts(chunks, metadatas=metadatas)

    def query(self, query, k=10):
        """Query the SQLiteVSS vector store."""
        return self.vector_store.similarity_search(query, k=k)
