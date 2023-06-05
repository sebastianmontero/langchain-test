from abc import ABC, abstractmethod
from pydantic import BaseModel, Extra,  PrivateAttr
import logging
from langchain.prompts.base import BasePromptTemplate
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain.text_splitter import TextSplitter, RecursiveCharacterTextSplitter
from langchain.base_language import BaseLanguageModel
from langchain.chains.summarize import load_summarize_chain
from typing import Any

logger = logging.getLogger(__name__)
logger.setLevel(level=logging.DEBUG)
logger.addHandler(logging.StreamHandler())


proposal_summary_prompt_template = """You are an expert proposal summarizer, you generate summaries that
enable anyone to understand the most important details of a proposal. Write a concise summary of the following:


"{text}"


CONCISE SUMMARY:"""
PROPOSAL_SUMMARY_PROMPT = PromptTemplate(
    template=proposal_summary_prompt_template, input_variables=["text"])


class BaseSummarizer(ABC, BaseModel):

    @abstractmethod
    def summarize(self, s: Document) -> str:
        """Summarize document"""


class Summarizer(BaseSummarizer):
    llm: BaseLanguageModel
    text_splitter: TextSplitter = RecursiveCharacterTextSplitter(
        chunk_size=3000, chunk_overlap=0)
    prompt: BasePromptTemplate
    verbose: bool = False
    _chain = PrivateAttr()

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.allow
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    def summarize(self, doc: Document) -> str:
        split_docs = self.text_splitter.split_documents([doc])
        args= {
            "llm": self.llm,
            "verbose": self.verbose,
        }
        if len(split_docs) == 1:
            args["prompt"] = self.prompt
        else:
            args["chain_type"] = "map_reduce"
            args["map_prompt"] = self.prompt
            args["combine_prompt"] = self.prompt

        chain = load_summarize_chain(**args)
        logger.debug("Summarizing...")
        return chain.run(split_docs)

    @classmethod
    def proposal_summarizer(cls, llm: BaseLanguageModel, verbose: bool, **data: Any) -> "Summarizer":
        return cls(
            llm=llm,
            verbose=verbose,
            prompt=PROPOSAL_SUMMARY_PROMPT,
            **data
        )
