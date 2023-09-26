from dotenv import load_dotenv
# from langchain.document_loaders import GoogleDriveLoader
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.callbacks import get_openai_callback
from langchain.schema import messages_from_dict
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from governance_common.chatbots import proposal_chatbot
import pinecone

import os


# Using the conversational retrieval chain with chat model(gpt-4), it takes care of querying the index and
# remembering chat history

def main():
    load_dotenv()

    # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    # PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    # PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
    # PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
    # PROPOSAL_NAMESPACE = os.getenv("PROPOSAL_NAMESPACE")
    # proposal_path = "polkadot/treasury_proposals/203"

    # embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    # pinecone.init(
    #     api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    #     environment=PINECONE_API_ENV  # next to api key in console
    # )    
    # vector_store = Pinecone.from_existing_index(index_name=PINECONE_INDEX_NAME, embedding=embeddings)
    # retriever = vector_store.as_retriever(search_kwargs={"namespace":PROPOSAL_NAMESPACE, "filter":{"governance_proposal_path": proposal_path},  "k":4})
    # llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-4")
    # qa_chain = load_qa_chain(llm=llm, chain_type="stuff")
    # question_generator_chain=LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
    # memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key="answer")
    # chain = ConversationalRetrievalChain(combine_docs_chain=qa_chain, retriever=retriever, return_source_documents=True, question_generator=question_generator_chain, verbose=True)

    chain = proposal_chatbot(proposal_path="polkadot/treasury_proposals/203", return_source_documents=True)
    while True:
        # Ask the user for their name
        question = input("Please ask a question (Type 'exit' to quit): ")
        # Check if the user wants to exit
        if question.lower() == "exit":
            break

        chat_history = [
            {
                "data":{ "content": "what is the proposal about"},
                "type": "human"
            },
            {
                "data":{ "content": "The Unit Masters Proposal is about onboarding people to web3 and the Polkadot ecosystem through education. The goal is to address the lack of skills and drive mainstream adoption by providing free, high-quality education to ensure equitability in the web3 space, with the Polkadot ecosystem as the first contact point for new joiners. The proposal includes a team of experts, a breakdown of costs, milestones, and expected outcomes, as well as partnerships and course curriculum details."},
                "type": "ai"
            }
        ]
        # chat_history = []
    
        chat_history = messages_from_dict(chat_history)

        with get_openai_callback() as cb:
            result = chain({"question": question, "chat_history": chat_history})
            print(f"Call back:\n {cb}")

        print(f"Result:\n {result}")
        print(f"Answer:\n {result['answer']}")
        # sprint(f"Sources:\n {sources}")




if __name__ == "__main__":
    main()

  