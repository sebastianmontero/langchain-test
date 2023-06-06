
from langchain.agents.agent import AgentExecutor
from langchain.vectorstores import VectorStore, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent, Tool, load_tools
from langchain_util.chains import ConcatenateChain
from langchain_util.tools import AgentAsTool
from typing import Optional
from langchain.base_language import BaseLanguageModel
from langchain.prompts import PromptTemplate
from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict
from typing import List, Optional

import pinecone 

import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
POLKADOT_WIKI_NAMESPACE = os.getenv("POLKADOT_WIKI_NAMESPACE")
PROPOSAL_NAMESPACE = os.getenv("PROPOSAL_NAMESPACE")
DB_URL = os.getenv("DB_URL")


RETRIEVAL_TOOL_TEMPALTE = PromptTemplate(
    input_variables=["source_description"],
    template="""Use this tool to answer user questions about {source_description}, you should pass
        in the user question related to the proposal with as little modification as possible.
        This tool can also be used for follow up questions from the user."""
    )



def governance_expert(model_name: str = "gpt-4", memory_window_size = 5, verbose: bool = False)  -> AgentExecutor:

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
        # memory=memory
        )

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
        namespace=POLKADOT_WIKI_NAMESPACE,
        verbose=verbose
    ))

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Proposals DB",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="proposals"
        ),
        namespace=PROPOSAL_NAMESPACE,
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

def multiagent_expert(model_name: str = "gpt-4", memory_window_size: Optional[int] = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

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
        document_contents='Information about polkadot on how it works, how to build on it and how to setup nodes and run the network',
        namespace=POLKADOT_WIKI_NAMESPACE,
        verbose=verbose
    ))

    metadata_field_info=[
        AttributeInfo(
            name="governance_proposal_path",
            description="The path of the proposal, the path format is: network_name/proposal_type/id", 
            type="string", 
        )
    ]

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Proposals Vector DB",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="the content of a specific proposal, objective, milestones, budget, team, etc. Make sure to include the proposal path in the question"
        ),
        namespace=PROPOSAL_NAMESPACE,
        document_contents='proposal details',
        metadata_field_info=metadata_field_info,
        verbose=verbose
    ))

    db = SQLDatabase.from_uri(DB_URL)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    tools.append(AgentAsTool(
        name="Proposals SQL DB",
        # description="""Use this tool to answer user questions about information related to proposals but not their content and about the proposals database structure, you should pass
        # in the user question related to the proposal with as little modification as possible.
        # This tool can also be used for follow up questions from the user.""",
        # description="""Use this tool to answer user questions about information related to proposals and the comments and reactions to them, you should pass
        # in the user question related to the proposal with as little modification as possible.
        # This tool can also be used for follow up questions from the user.""",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="information related to proposals, comments of the proposals and reactions to the proposals"
        ),
        agent=sql_agent
    ))

    memory = None
    if memory_window_size:
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
        memory=memory
    )

    # sys_msg = (
    #     "You are an expert on the polkadot blockchain, you are able to answer user questions "
    #     "in an easy to understand way always providing the important details."
    #     "When users ask information you refer to the relevant tools "
    #     "when needed, allowing you to answer questions about polkadot.\n"
    #     "When external information is used you MUST add your sources to the end "
    #     "of responses, the format should be: SOURCES:[s1,s2,...]"
    # )

    sys_msg = (
        "You are an expert on the polkadot blockchain, you are able to answer user questions "
        "in an easy to understand way always providing the important details."
        "When users ask information you refer to the relevant tools "
        "when needed, allowing you to answer questions about polkadot.\n"
        "When external information is used you MUST add your sources to the end "
        "of responses"
    )

    prompt = agent.agent.create_prompt(
        system_message=sys_msg,
        tools=tools
    )
    agent.agent.llm_chain.prompt = prompt

    return agent

def multiagent_expert_history_through_memory(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:
    """Example that shows how to pre-populate the agent memory with history"""
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
        document_contents='Information about polkadot on how it works, how to build on it and how to setup nodes and run the network',
        namespace=POLKADOT_WIKI_NAMESPACE,
        verbose=verbose
    ))

    metadata_field_info=[
        AttributeInfo(
            name="governance_proposal_path",
            description="The path of the proposal, the path format is: network_name/proposal_type/id", 
            type="string", 
        )
    ]

    tools.append(_get_retrieval_tool(
        llm=llm,
        vector_store=vector_store,
        name="Proposals Vector DB",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="the content of a specific proposal, objective, milestones, budget, team, etc. Make sure to include the proposal path in the question"
        ),
        namespace=PROPOSAL_NAMESPACE,
        document_contents='proposal details',
        metadata_field_info=metadata_field_info,
        verbose=verbose
    ))

    db = SQLDatabase.from_uri(DB_URL)
    toolkit = SQLDatabaseToolkit(db=db, llm=llm)

    sql_agent = create_sql_agent(
        llm=llm,
        toolkit=toolkit,
        verbose=True
    )

    tools.append(AgentAsTool(
        name="Proposals SQL DB",
        # description="""Use this tool to answer user questions about information related to proposals but not their content and about the proposals database structure, you should pass
        # in the user question related to the proposal with as little modification as possible.
        # This tool can also be used for follow up questions from the user.""",
        # description="""Use this tool to answer user questions about information related to proposals and the comments and reactions to them, you should pass
        # in the user question related to the proposal with as little modification as possible.
        # This tool can also be used for follow up questions from the user.""",
        description=RETRIEVAL_TOOL_TEMPALTE.format(
            source_description="information related to proposals, comments of the proposals and reactions to the proposals"
        ),
        agent=sql_agent
    ))

    chat_history = [
        {
            "data":{ "content": "what is the purpose of the proposal with path polkadot/treasury_proposals/231"},
            "type": "human"
        },
        {
            "data":{ "content": "The purpose of the proposal with path polkadot/treasury_proposals/231 is to provide 6-month maintenance funding for The Kusamarian, a platform that serves token holders by setting a standard for the propagation of vital information and narrative across the Polkadot network. The Kusamarian focuses on software development, community building, and realistic expectations, with an emphasis on fair, truthful, accurate reporting, and journalistic independence. The funding will support the continuation and improvement of their various programs, including Alpha Shots, PNinja on the Road, and Attempts at Governance. SOURCES: [https://docs.google.com/document/d/12b5dPQaPvu-59j2Bf4LoJYRwV_xjIM95t8kCHpMhTPY/edit#bookmark=kix.k2t917l25ohw]"},
            "type": "ai"
        }
    ]
    
    chat_history = messages_from_dict(chat_history)

    memory = ConversationBufferWindowMemory(
        # important to align with agent prompt (below)
        chat_memory=ChatMessageHistory(messages=chat_history),
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
        memory=memory
        )

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

# def multiagent_expert_3(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
#     PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
#     DB_URL = os.getenv("DB_URL")

#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     pinecone.init(
#         api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#         environment=PINECONE_API_ENV  # next to api key in console
#     )
#     vector_store = Pinecone.from_existing_index(
#         index_name=PINECONE_INDEX_NAME, embedding=embeddings)
#     llm = ChatOpenAI(
#         temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    
#     tools = load_tools(tool_names=['llm-math'], llm=llm)

#     tools.append(_get_retrieval_tool(
#         llm=llm,
#         vector_store=vector_store,
#         name="Polkadot Wiki",
#         description=RETRIEVAL_TOOL_TEMPALTE.format(
#             source_description="polkadot, it has information on how it works, how to build on it and how to setup nodes and run the network"
#         ),
#         filter={
#             "type":"docs"
#         },
#         verbose=verbose
#     ))

#     # tools.append(_get_retrieval_tool(
#     #     llm=llm,
#     #     vector_store=vector_store,
#     #     name="Proposals Vector DB",
#     #     description=RETRIEVAL_TOOL_TEMPALTE.format(
#     #         source_description="the content of proposals"
#     #     ),
#     #     filter={
#     #         "type":"proposals"
#     #     },
#     #     verbose=verbose
#     # ))

#     db = SQLDatabase.from_uri(DB_URL)
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#     music_agent = create_sql_agent(
#         llm=llm,
#         toolkit=toolkit,
#         verbose=True
#     )

#     tools.append(AgentAsTool(
#         name="Proposals SQL DB",
#         # description="""Use this tool to answer user questions about information related to proposals but not their content and about the proposals database structure, you should pass
#         # in the user question related to the proposal with as little modification as possible.
#         # This tool can also be used for follow up questions from the user.""",
#         description="""Use this tool to answer user questions about information related to proposals and the comments and reactions to them, you should pass
#         in the user question related to the proposal with as little modification as possible.
#         This tool can also be used for follow up questions from the user.""",
#         agent=music_agent
#     ))

#     memory = ConversationBufferWindowMemory(
#         # important to align with agent prompt (below)
#         memory_key="chat_history",
#         k=memory_window_size,
#         return_messages=True
#     )

#     agent = initialize_agent(
#         agent="chat-conversational-react-description",
#         llm=llm,
#         tools=tools,
#         verbose=verbose,
#         max_iterations=2,
#         early_stopping_method="generate",
#         memory=memory)

#     sys_msg = (
#         "You are an expert on the polkadot blockchain, you are able to answer user questions "
#         "in an easy to understand way always providing the important details."
#         "When users ask information you refer to the relevant tools "
#         "when needed, allowing you to answer questions about polkadot.\n"
#         "When external information is used you MUST add your sources to the end "
#         "of responses, the format should be: SOURCES:[s1,s2,...]"
#     )

#     prompt = agent.agent.create_prompt(
#         system_message=sys_msg,
#         tools=tools
#     )
#     agent.agent.llm_chain.prompt = prompt

#     return agent

# def multiagent_expert_2(model_name: str = "gpt-4", memory_window_size = 5, similarity_threshold:float = 0.76, verbose: bool = False)  -> AgentExecutor:

#     OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
#     PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
#     PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
#     PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

#     embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
#     pinecone.init(
#         api_key=PINECONE_API_KEY,  # find at app.pinecone.io
#         environment=PINECONE_API_ENV  # next to api key in console
#     )
#     vector_store = Pinecone.from_existing_index(
#         index_name=PINECONE_INDEX_NAME, embedding=embeddings)
#     llm = ChatOpenAI(
#         temperature=0, openai_api_key=OPENAI_API_KEY, model_name=model_name)
    
#     tools = load_tools(tool_names=['llm-math'], llm=llm)

#     tools.append(_get_retrieval_tool(
#         llm=llm,
#         vector_store=vector_store,
#         name="Polkadot Wiki",
#         description=RETRIEVAL_TOOL_TEMPALTE.format(
#             source_description="polkadot, it has information on how it works, how to build on it and how to setup nodes and run the network"
#         ),
#         filter={
#             "type":"docs"
#         },
#         verbose=verbose
#     ))

#     tools.append(_get_retrieval_tool(
#         llm=llm,
#         vector_store=vector_store,
#         name="Proposals DB",
#         description=RETRIEVAL_TOOL_TEMPALTE.format(
#             source_description="proposals"
#         ),
#         filter={
#             "type":"proposals"
#         },
#         verbose=verbose
#     ))

#     data_path = os.path.join(os.path.dirname(__file__), "../data/Chinook.db")
#     print(f"Data path: {data_path}")
#     db = SQLDatabase.from_uri(f"sqlite:///{data_path}")
#     toolkit = SQLDatabaseToolkit(db=db, llm=llm)

#     music_agent = create_sql_agent(
#         llm=llm,
#         toolkit=toolkit,
#         verbose=True
#     )

#     tools.append(AgentAsTool(
#         name="Music Store DB",
#         description="""Use this tool to answer user questions about the music store and the music store database structure, you should pass
#         in the user question related to the proposal with as little modification as possible.
#         This tool can also be used for follow up questions from the user.""",
#         agent=music_agent
#     ))

#     memory = ConversationBufferWindowMemory(
#         # important to align with agent prompt (below)
#         memory_key="chat_history",
#         k=memory_window_size,
#         return_messages=True
#     )

#     agent = initialize_agent(
#         agent="chat-conversational-react-description",
#         llm=llm,
#         tools=tools,
#         verbose=verbose,
#         max_iterations=2,
#         early_stopping_method="generate",
#         memory=memory)

#     sys_msg = (
#         "You are an expert on the polkadot blockchain, you are able to answer user questions "
#         "in an easy to understand way always providing the important details."
#         "When users ask information you refer to the relevant tools "
#         "when needed, allowing you to answer questions about polkadot.\n"
#         "When external information is used you MUST add your sources to the end "
#         "of responses, the format should be: SOURCES:[s1,s2,...]"
#     )

#     prompt = agent.agent.create_prompt(
#         system_message=sys_msg,
#         tools=tools
#     )
#     agent.agent.llm_chain.prompt = prompt

#     return agent


def  _get_retrieval_tool(llm: Optional[BaseLanguageModel], vector_store: VectorStore, name: str, description:  str, namespace: Optional[str], document_contents: str, similarity_threshold: float = 0.76, metadata_field_info: Optional[List[AttributeInfo]] = None, verbose=False):
    # retriever = vector_store.as_retriever(search_type="similarity_score_threshold", search_kwargs={"score_threshold":similarity_threshold,  "k":10})

    if metadata_field_info:
        retriever = SelfQueryRetriever.from_llm(
            llm=llm,
            vectorstore=vector_store,
            document_contents=document_contents,
            metadata_field_info=metadata_field_info,
            search_kwargs={
                "namespace": namespace
            },
            verbose=verbose
        )
    else:
        retriever = vector_store.as_retriever(search_kwargs={"namespace":namespace,  "k":4})

    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever, verbose=verbose)
    tool_chain = ConcatenateChain(input_chain=chain, keys=["answer", "sources"], output_key="answer", verbose=verbose)

    return Tool(
        func=tool_chain.run,
        name=name,
        description=description,
        verbose=verbose
    )
