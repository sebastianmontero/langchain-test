from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra
import logging
from langchain.vectorstores import VectorStore
from langchain.embeddings.base import Embeddings
from langchain.schema import Document
from langchain_util.text_splitter import TextSplitterWithContext, RecursiveCharacterTextSplitterWithContext
from typing import Type

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class BaseDocumentStore(ABC, BaseModel):
    
    @abstractmethod
    def store(self, s: Document) -> bool:
        """Process and store document"""

class VectorDocumentStore(BaseDocumentStore):
    vector_store: Type[VectorStore]
    index_name: str
    namespace: str
    embeddings: Embeddings
    text_splitter: TextSplitterWithContext = RecursiveCharacterTextSplitterWithContext(chunk_size=2000, chunk_overlap=0)
    
    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True


    def store(self, doc: Document):
        """Process and store document"""
        split_docs = self.text_splitter.split_documents([doc])
        total_docs = len(split_docs)
        for i, sd in enumerate(split_docs):
            # sd.metadata["source"] = f"{sd.metadata['source']}({i}/{total_docs})"
            sd.metadata["source"] = f"{sd.metadata['source']}"
        logger.debug("Storing chunks in vector database...")
        self.vector_store.from_documents(documents=split_docs, embedding=self.embeddings, index_name=self.index_name, namespace=self.namespace)
        
