
from langchain.agents.agent import AgentExecutor
from langchain.vectorstores import VectorStore, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool, load_tools
from langchain_util.chains import ConcatenateChain
from langchain_util.tools import AgentAsTool
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import EmbeddingsFilter
from typing import Optional
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase


import pinecone 

import os


RETRIEVAL_TOOL_TEMPALTE = PromptTemplate(
    input_variables=["source_description"],
    template="""Use this tool to answer user questions about {source_description}, you should pass
        in the user question related to the proposal with as little modification as possible.
        This tool can also be used for follow up questions from the user."""
    )



def governance_expert(model_name: str = "gpt-4", memory_window_size = 5, verbose: bool = False)  -> AgentExecutor:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), verbose=verbose)
    tool_chain = ConcatenateChain(input_chain=chain, keys=["answer", "sources"], output_key="answer")
    tools = load_tools(tool_names=['llm-math'], llm=llm)

    # tool_description = """Use this tool to answer user questions about proposals, you should pass
    #     in the user question related to the proposal with as little modification as possible.
    #     This tool can also be used for follow up questions from the user."""
    tool_description = """Use this tool to answer user questions about proposals, you should pass
        in the user question related to the proposal with as little modification as possible.
        This tool can also be used for follow up questions from the user."""
    tools.append(Tool(
        func=tool_chain.run,
        name="Polkadot Proposal DB",
        description=tool_description,
    ))

    memory = ConversationBufferWindowMemory(
        # important to align with agent prompt (below)
        memory_key="chat_history",
        k=memory_window_size,
        return_messages=True
    )

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        llm=llm,
        tools=tools,
        verbose=True,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory)

    sys_msg = (
        "You are an expert on on-chain proposals, you are able to answer user questions "
        "in an easy to understand way always providing the important details."
        "When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about the proposals.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses, the format should be: SOURCES:[s1,s2,...]"
    )

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent


def polkadot_expert(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    
    retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":0.5,  "k":10})
    # if compress_context:
    #     compressor = EmbeddingsFilter.from_llm(llm)
    #     retriever = ContextualCompressionRetriever(base_compressor=compressor, base_retriever=retriever, verbose=verbose)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=verbose)
    tool_chain = ConcatenateChain(input_chain=chain, keys=["answer", "sources"], output_key="answer", verbose=verbose)
    tools = load_tools(tool_names=['llm-math'], llm=llm)

    # tool_description = """Use this tool to answer user questions about proposals, you should pass
    #     in the user question related to the proposal with as little modification as possible.
    #     This tool can also be used for follow up questions from the user."""
    tool_description = """Use this tool to answer user questions about polkadot, you should pass
        in the user question related to polkadot with as little modification as possible.
        This tool can also be used for follow up questions from the user."""
    # tool_description = """Use this tool to answer user questions about polkadot.
    #     This tool can also be used for follow up questions from the user."""
    tools.append(Tool(
        func=tool_chain.run,
        name="Polkadot DB",
        description=tool_description,
        verbose=verbose
    ))

    memory = ConversationBufferWindowMemory(
        # important to align with agent prompt (below)
        memory_key="chat_history",
        k=memory_window_size,
        return_messages=True
    )

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        llm=llm,
        tools=tools,
        verbose=verbose,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory)

    sys_msg = (
        "You are an expert on the polkadot blockchain, you are able to answer user questions "
        "in an easy to understand way always providing the important details."
        "When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about polkadot.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses, the format should be: SOURCES:[s1,s2,...]"
    )

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent


def general_expert(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    
    tools = load_tools(tool_names=['llm-math'], llm=llm)

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Polkadot Wiki",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="polkadot, it has information on how it works, how to build on it and how to setup nodes and run the network"
        ),
        filter={
            "type":"docs"
        },
        verbose=verbose
    ))

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Proposals DB",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="proposals"
        ),
        filter={
            "type":"proposals"
        },
        verbose=verbose
    ))

    memory = ConversationBufferWindowMemory(
        # important to align with agent prompt (below)
        memory_key="chat_history",
        k=memory_window_size,
        return_messages=True
    )

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        llm=llm,
        tools=tools,
        verbose=verbose,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory)

    sys_msg = (
        "You are an expert on the polkadot blockchain, you are able to answer user questions "
        "in an easy to understand way always providing the important details."
        "When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about polkadot.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses, the format should be: SOURCES:[s1,s2,...]"
    )

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent

def multiagent_expert(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )
    vector_store = Pinecone.from_existing_index(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    
    tools = load_tools(tool_names=['llm-math'], llm=llm)

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Polkadot Wiki",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="polkadot, it has information on how it works, how to build on it and how to setup nodes and run the network"
        ),
        filter={
            "type":"docs"
        },
        verbose=verbose
    ))

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Proposals DB",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="proposals"
        ),
        filter={
            "type":"proposals"
        },
        verbose=verbose
    ))

    data_path = os.path.join(os.path.dirname(__file__), "../data/Chinook.db")
    print(f"Data path: {data_path}")
    db = SQLDatabase.from_uri(f"sqlite:///{data_path}")
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    music_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    tools.append(AgentAsTool(
        name="Music Store DB",
        description="""Use this tool to answer user questions about the music store and the music store database structure, you should pass
        in the user question related to the proposal with as little modification as possible.
        This tool can also be used for follow up questions from the user.""",
        agent=music_agent
    ))

    memory = ConversationBufferWindowMemory(
        # important to align with agent prompt (below)
        memory_key="chat_history",
        k=memory_window_size,
        return_messages=True
    )

    agent = initialize_agent(
        agent="chat-conversational-react-description",
        llm=llm,
        tools=tools,
        verbose=verbose,
        max_iterations=2,
        early_stopping_method="generate",
        memory=memory)

    sys_msg = (
        "You are an expert on the polkadot blockchain, you are able to answer user questions "
        "in an easy to understand way always providing the important details."
        "When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about polkadot.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses, the format should be: SOURCES:[s1,s2,...]"
    )

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent


def  _get_retrieval_tool(llm: Optional[BaseLanguageModel], vector_store: VectorStore, name: str, description:  str, filter: Optional[dict], similarity_threshold: float = 0.76, verbose=False):
    # retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":similarity_threshold,  "k":10})
    retriever = vector_store.as_retriever(search_kwargs={"filter":filter,  "k":4})
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=verbose)
    tool_chain = ConcatenateChain(input_chain=chain, keys=["answer", "sources"], output_key="answer", verbose=verbose)

    return Tool(
        func=tool_chain.run,
        name=name,
        description=description,
        verbose=verbose
    )
