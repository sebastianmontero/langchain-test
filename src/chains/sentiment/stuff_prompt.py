# flake8: noqa
from langchain.chains.prompt_selector import ConditionalPromptSelector, is_chat_model
from langchain.prompts import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from chains.sentiment.parser import PARSER

format_instructions = PARSER.get_format_instructions()
prompt_template = """Determine the sentiment of the following STATEMENT and provide a score on how confident you are of this.

{format_instructions}

STATEMENT:
{context}

YOUR RESPONSE:"""
PROMPT = PromptTemplate(
    template=prompt_template, input_variables=[
        "format_instructions", "context"]
)
PROMPT = PROMPT.partial(format_instructions=format_instructions)

system_template = """You are a sentiment analyst expert, you determine the sentiment of the user statements and provide a score from [0,1] on how confident you are of this.

{format_instructions}"""

system_prompt = PromptTemplate(
    template=system_template, input_variables=["format_instructions"])
system_prompt = system_prompt.partial(format_instructions=format_instructions)
messages = [
    SystemMessagePromptTemplate(prompt=system_prompt),
    HumanMessagePromptTemplate.from_template("{context}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


PROMPT_SELECTOR = ConditionalPromptSelector(
    default_prompt=PROMPT, conditionals=[(is_chat_model, CHAT_PROMPT)]
)
