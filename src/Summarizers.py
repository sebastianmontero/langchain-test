from abc import ABC, abstractmethod
from pydantic import BaseModel,  PrivateAttr
import logging
from langchain.schema import Document
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.base_language import BaseLanguageModel
from langchain.chains.summarize import load_summarize_chain

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


class BaseSummarizer(ABC, BaseModel):
    
    @abstractmethod
    def summarize(self, s: Document) -> str:
        """Summarize document"""

class Summarizer(BaseSummarizer):
    llm: BaseLanguageModel
    text_splitter: TextSplitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    chain_type: str = "map_reduce"
    verbose: bool = False
    _chain = PrivateAttr()

    def __init__(self, **data):
        super().__init__(**data)
        self._chain = load_summarize_chain(llm=self.llm, chain_type=self.chain_type, verbose=self.verbose)

    @abstractmethod
    def summarize(self, doc: Document) -> str:
        split_docs = self.text_splitter.split_documents([doc])
        return self._chain.run(split_docs)
        
