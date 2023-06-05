from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
# from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

import pinecone

import os


# Using the conversational retrieval chain with chat model(gpt-4), it takes care of querying the index and
# remembering chat history

_template = """Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question, in its original language.
If you have the proposal path make sure to add this to the follow up question, this is the best way to identify the proposal so no other identifying information is
required.
Chat History:
{chat_history}
Follow Up Input: {question}
Standalone question:"""
CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)


system_template = """Use the following pieces of context to answer the users question. 
If you don't know the answer, just say that you don't know, don't try to make up an answer.
If in the context remember to mention the proposal paths in the answer
----------------
{context}"""
messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
CHAT_PROMPT = ChatPromptTemplate.from_messages(messages)


def main():
    load_dotenv()

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    PROPOSAL_NAMESPACE = os.getenv("PROPOSAL_NAMESPACE")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )    
    vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    metadata_field_info=[
        AttributeInfo(
            name="governance_proposal_path",
            description="The path of the proposal, the path format is: network_name/proposal_type/id", 
            type="string", 
        ),
        AttributeInfo(
            name="network",
            description="The network the propsal belongs to", 
            type="string", 
        )
    ]
    self_query_retriever= SelfQueryRetriever.from_llm(
        llm=llm,
        vectorstore=vector_store,
        document_contents="Proposals",
        metadata_field_info=metadata_field_info,
        search_kwargs={
            "namespace": PROPOSAL_NAMESPACE
        },
        verbose=True

    )
    qa_chain = load_qa_chain(llm=llm, chain_type="stuff", prompt=CHAT_PROMPT)
    question_generator_chain=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    chain = ConversationalRetrievalChain(combine_docs_chain=qa_chain, retriever=self_query_retriever, return_source_documents=True,  memory=memory, question_generator=question_generator_chain, verbose=True)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break
        
        answer = chain(question)

        print(f"Result:\n {answer}")
        print(f"Answer:\n {answer['answer']}")
        # sprint(f"Sources:\n {sources}")




if __name__ == "__main__":
    main()

  