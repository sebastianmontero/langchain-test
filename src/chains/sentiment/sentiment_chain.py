from abc import abstractmethod
from typing import Any, Dict, List, Optional

from pydantic import Extra, Field, root_validator

from langchain.base_language import BaseLanguageModel
from langchain.callbacks.manager import (
    AsyncCallbackManagerForChainRun,
    CallbackManagerForChainRun,
)
from langchain.chains.base import Chain
from langchain.chains.combine_documents.base import BaseCombineDocumentsChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from chains.sentiment.stuff_prompt import PROMPT_SELECTOR
from langchain.prompts import PromptTemplate
from langchain.schema import BaseOutputParser
from chains.sentiment.parser import PARSER



class SentimentChain(Chain):
    output_parser: BaseOutputParser
    """Object to parse the output returned by the chain"""
    combine_documents_chain: BaseCombineDocumentsChain
    """Chain to use to combine the documents."""
    input_key: str = "input_documents"  #: :meta private:

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True
        allow_population_by_field_name = True

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.

        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.

        :meta private:
        """
        return ["sentiment", "confidence"]

    @classmethod
    def from_llm(
        cls,
        llm: BaseLanguageModel,
        prompt: Optional[PromptTemplate] = None,
        **kwargs: Any,
    ) -> "SentimentChain":
        """Initialize from LLM."""
        _prompt = prompt or PROMPT_SELECTOR.get_prompt(llm)
        llm_chain = LLMChain(llm=llm, prompt=_prompt)
        document_prompt = PromptTemplate(
            input_variables=["page_content"], template="{page_content}"
        )
        combine_documents_chain = StuffDocumentsChain(
            llm_chain=llm_chain,
            document_variable_name="context",
            document_prompt=document_prompt,
        )

        return cls(output_parser=PARSER, combine_documents_chain=combine_documents_chain, **kwargs)

    def _call(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[CallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run llm on input documents.

        Example:
        .. code-block:: python

        res = chain({'input_documents': docs})
        sentiment, confidence = res['sentiment'], res['confidence']
        """
        _run_manager = run_manager or CallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]

        result = self.combine_documents_chain.run(
            input_documents=docs, callbacks=_run_manager.get_child()
        )
        return self.output_parser.parse(result)


    async def _acall(
        self,
        inputs: Dict[str, Any],
        run_manager: Optional[AsyncCallbackManagerForChainRun] = None,
    ) -> Dict[str, Any]:
        """Run llm on input documents.

        Example:
        .. code-block:: python

        res = chain({'input_documents': docs})
        sentiment, confidence = res['sentiment'], res['confidence']
        """
        _run_manager = run_manager or AsyncCallbackManagerForChainRun.get_noop_manager()
        docs = inputs[self.input_key]
        result = await self.combine_documents_chain.arun(
            input_documents=docs, callbacks=_run_manager.get_child()
        )
        return self.output_parser.parse(result)
