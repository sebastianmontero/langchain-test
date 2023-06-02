from abc import ABC, abstractmethod
from pydantic import BaseModel
import logging
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_util.text_splitter import TextSplitterWithContext, RecursiveCharacterTextSplitterWithContext

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class BaseDocumentStore(ABC, BaseModel):
    
    @abstractmethod
    def store(self, s: Document) -> bool:
        """Process and store document"""

class VectorDocumentStore(BaseDocumentStore):
    vector_store: VectorStore
    index_name: str
    embeddings: Embeddings
    text_splitter: TextSplitterWithContext = RecursiveCharacterTextSplitterWithContext(chunk_size=2000, chunk_overlap=0)
    

    def store(self, doc: Document):
        """Process and store document"""
        split_docs = self.text_splitter.split_documents([doc])
        total_docs = len(split_docs)
        for i, sd in enumerate(split_docs):
            sd.metadata["source"] = f"{sd.metadata['source']}[{i}/{total_docs}]"
        self.vector_store.from_documents(documents=split_docs, embedding=self.embeddings, index_name=self.index_name)
        
