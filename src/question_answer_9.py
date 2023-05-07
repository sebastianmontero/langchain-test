from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.agents import initialize_agent, Tool, load_tools
from langchain.callbacks import get_openai_callback
from chains import ConcatenateChain


import pinecone

import os


# Using the conversational agent that uses retrieval qa as a tool, it takes care of querying the index

def main():
    load_dotenv()

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
        temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm, chain_type="stuff", retriever=vector_store.as_retriever(), verbose=True)
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
        k=5,
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

    # prompt = agent.agent.create_prompt(tools,
    # prefix="""Answer the following questions as best you can, if you don't know the answer just say "I don't know".
    # To answer some of the questions, you might need to perform multiple steps to collect the information you need to provide the answer.
    # You have access to the following tools:""",)
    # agent.agent.llm_chain.prompt = prompt

    print(agent.agent.llm_chain.prompt)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break

        answer = run_with_token_count(agent, question)

        print(f"Answer:\n {answer}")
        # sprint(f"Sources:\n {sources}")


def run_with_token_count(agent, query):
    with get_openai_callback() as cb:
        result = agent(query)
        print(f'Spent a total of {cb.total_tokens} tokens')

    return result


if __name__ == "__main__":
    main()
