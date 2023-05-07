from typing import Dict, List, Optional
from langchain.chains.base import Chain
from pydantic import Extra

class ConcatenateChain(Chain):
    """
    A custom chain that takes the output of another chain and concatenates
    the different values of the output into a single string and returns it.
    """
    input_chain: Chain
    """The input chain whose output will be concatenated."""
    
    keys: Optional[List[str]]
    """An optional list of keys to include in the concatenation."""
    
    output_key: str
    """The name of the key in the final output."""

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        return self.input_chain.input_keys

    @property
    def output_keys(self) -> List[str]:
        return [self.output_key]
    
    @property
    def _chain_type(self) -> str:
        return "concatenate_chain"

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        input_chain_output = self.input_chain(inputs)
        if self.keys:
            items = [(key, input_chain_output[key]) for key in self.keys]
        else:
            items = input_chain_output.items()
        result = " ".join([f"{key}: {value}" for key, value in items])
        return {self.output_key: result}